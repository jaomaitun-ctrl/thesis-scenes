#include <ViscoOgden/ViscoOgdenFEMForceField.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace viscoogden
{

using sofa::Index;
using Real = ViscoOgdenFEMForceField::Real;
using Vec3 = ViscoOgdenFEMForceField::Vec3;
using Mat3 = ViscoOgdenFEMForceField::Mat3;

namespace
{
Mat3 outer(const Vec3& a, const Vec3& b)
{
    Mat3 m;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m(i, j) = a[i] * b[j];
    return m;
}
Real frob(const Mat3& m)
{
    Real s = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            s += m(i, j) * m(i, j);
    return std::sqrt(s);
}
} // namespace

ViscoOgdenFEMForceField::ViscoOgdenFEMForceField()
    : d_mu(initData(&d_mu, VecReal{1000.0}, "mu", "Long-term Ogden moduli mu_p [Pa]; small-strain shear modulus = sum mu_p"))
    , d_alpha(initData(&d_alpha, VecReal{-20.0}, "alpha", "Ogden exponents alpha_p (same length as mu, non-zero)"))
    , d_bulkModulus(initData(&d_bulkModulus, Real(1e5), "bulkModulus", "Bulk modulus kappa [Pa]; U = kappa/2 (ln J)^2"))
    , d_beta(initData(&d_beta, VecReal{}, "beta", "Prony factors beta_i >= 0 (instantaneous = (1 + sum beta) x long-term)"))
    , d_tau(initData(&d_tau, VecReal{}, "tau", "Prony relaxation times tau_i [s] (same length as beta)"))
    , d_checkTangent(initData(&d_checkTangent, false, "checkTangent", "Compare analytic and finite-difference element stiffness once (log)"))
    , d_viscous(initData(&d_viscous, true, "viscous", "If false, internal variables are frozen at zero (hyperelastic with long-term moduli)"))
    , d_internalStressNorm(initData(&d_internalStressNorm, "internalStressNorm", "Frobenius norm of sum_i Q_i per tetrahedron [Pa] (output)"))
    , d_stress(initData(&d_stress, "stress", "Total 2nd Piola-Kirchhoff stress per tetrahedron [Pa] (output)"))
    , d_elasticEnergy(initData(&d_elasticEnergy, Real(0), "elasticEnergy", "Long-term strain energy of the mesh [J] (output)"))
    , l_topology(initLink("topology", "Link to the tetrahedral topology (default: found in the context)"))
{
    d_internalStressNorm.setReadOnly(true);
    d_stress.setReadOnly(true);
    d_elasticEnergy.setReadOnly(true);
    this->f_listening.setValue(true);
}

ViscoOgdenFEMForceField::Material ViscoOgdenFEMForceField::material() const
{
    Material m;
    m.mu = d_mu.getValue();
    m.alpha = d_alpha.getValue();
    m.kappa = d_bulkModulus.getValue();
    m.g = 1;
    if (d_viscous.getValue())
    {
        const auto& beta = d_beta.getValue();
        const auto& tau = d_tau.getValue();
        const Real dt = this->getContext()->getDt();
        for (size_t i = 0; i < beta.size(); ++i)
            m.g += beta[i] * std::exp(-dt / (2 * tau[i]));
    }
    return m;
}

void ViscoOgdenFEMForceField::init()
{
    ForceField::init();
    if (l_topology.empty())
        l_topology.set(this->getContext()->getMeshTopologyLink());
    m_topology = l_topology.get();
    if (!m_topology || m_topology->getNbTetrahedra() == 0)
    {
        msg_error() << "No tetrahedral topology found.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    const auto& mu = d_mu.getValue();
    const auto& alpha = d_alpha.getValue();
    if (mu.empty() || mu.size() != alpha.size()
        || std::any_of(alpha.begin(), alpha.end(), [](Real a) { return std::abs(a) < 1e-12; }))
    {
        msg_error() << "mu and alpha must have the same, non-zero length, with alpha_p != 0.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    const auto& beta = d_beta.getValue();
    const auto& tau = d_tau.getValue();
    if (beta.size() != tau.size()
        || std::any_of(tau.begin(), tau.end(), [](Real t) { return !(t > 0); })
        || std::any_of(beta.begin(), beta.end(), [](Real b) { return b < 0; }))
    {
        msg_error() << "beta and tau must have the same length, with beta_i >= 0 and tau_i > 0.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    reinit();
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void ViscoOgdenFEMForceField::reinit()
{
    if (!m_topology)
        return;
    const VecCoord& X = this->mstate->read(sofa::core::vec_id::read_access::restPosition)->getValue();
    const auto& tets = m_topology->getTetrahedra();
    const size_t nTerms = d_beta.getValue().size();
    m_elements.assign(tets.size(), Element{});
    Real Vtot = 0;
    for (size_t t = 0; t < tets.size(); ++t)
    {
        Element& e = m_elements[t];
        for (int a = 0; a < 4; ++a)
            e.idx[a] = tets[t][a];
        Mat3 Dm;   // columns X_a - X_0
        for (int a = 1; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                Dm(i, a - 1) = X[e.idx[a]][i] - X[e.idx[0]][i];
        const Real det = sofa::type::determinant(Dm);
        e.V0 = std::abs(det) / 6;
        if (e.V0 < 1e-30)
        {
            msg_warning() << "Degenerate tetrahedron " << t << " ignored.";
            continue;
        }
        Mat3 Dinv;
        if (!sofa::type::invertMatrix(Dinv, Dm)) { e.V0 = 0; continue; }
        Vec3 sum;
        for (int a = 1; a < 4; ++a)
        {
            for (int j = 0; j < 3; ++j)
                e.dN[a][j] = Dinv(a - 1, j);
            sum += e.dN[a];
        }
        e.dN[0] = -sum;
        e.Q.assign(nTerms, Mat3());
        e.valid = true;
        Vtot += e.V0;
    }
    m_tangentChecked = false;
    d_internalStressNorm.setValue(VecReal(tets.size(), 0));
    d_stress.setValue(sofa::type::vector<Mat3>(tets.size(), Mat3()));
    msg_info() << tets.size() << " tetrahedra, rest volume " << Vtot << ", " << d_mu.getValue().size()
               << " Ogden term(s), " << nTerms << " Prony term(s)";
}

void ViscoOgdenFEMForceField::reset()
{
    for (auto& e : m_elements)
    {
        std::fill(e.Q.begin(), e.Q.end(), Mat3());
        e.SisoPrev = Mat3();
        e.H = Mat3();
    }
    d_internalStressNorm.setValue(VecReal(m_elements.size(), 0));
}

Mat3 ViscoOgdenFEMForceField::deformationGradient(const Element& e, const VecCoord& x) const
{
    Mat3 F;
    for (int a = 0; a < 4; ++a)
        F += outer(x[e.idx[a]], e.dN[a]);
    return F;
}

ViscoOgdenFEMForceField::PointState ViscoOgdenFEMForceField::evaluate(const Material& m, const Mat3& F, const Mat3& H)
{
    PointState p;
    p.F = F;
    p.J = sofa::type::determinant(F);
    if (!(p.J > 1e-8) || !std::isfinite(p.J))
        return p;
    const Mat3 C = F.transposed() * F;
    Eigen::Matrix3d Ce;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Ce(i, j) = C(i, j);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Ce);
    for (int a = 0; a < 3; ++a)
    {
        p.c[a] = std::max(es.eigenvalues()[a], 1e-12);
        for (int i = 0; i < 3; ++i)
            p.N(i, a) = es.eigenvectors()(i, a);
    }
    const Real lnJ = std::log(p.J);
    Vec3 siso, svol;
    Mat3 Aiso, Avol;
    Real W = m.kappa / 2 * lnJ * lnJ;
    for (size_t t = 0; t < m.mu.size(); ++t)
    {
        const Real mu = m.mu[t], al = m.alpha[t];
        const Real Jfac = std::pow(p.J, -al / 3);
        Vec3 lb;   // isochoric stretch^alpha
        for (int a = 0; a < 3; ++a)
            lb[a] = Jfac * std::pow(p.c[a], al / 2);
        const Real sum = lb[0] + lb[1] + lb[2];
        Vec3 dev;
        for (int a = 0; a < 3; ++a)
            dev[a] = lb[a] - sum / 3;
        W += 2 * mu / (al * al) * (sum - 3);
        for (int b = 0; b < 3; ++b)
        {
            siso[b] += 2 * mu / al * dev[b] / p.c[b];
            for (int d = 0; d < 3; ++d)
            {
                const Real dbd = (b == d) ? 1 : 0;
                Aiso(b, d) += 2 * mu / al / p.c[b]
                              * (-dbd * dev[b] / p.c[b]
                                 + al / (2 * p.c[d]) * (lb[b] * (dbd - Real(1) / 3) - dev[d] / 3));
            }
        }
    }
    for (int b = 0; b < 3; ++b)
    {
        svol[b] = m.kappa * lnJ / p.c[b];
        for (int d = 0; d < 3; ++d)
            Avol(b, d) = m.kappa * (1 / (2 * p.c[b] * p.c[d]) - ((b == d) ? lnJ / (p.c[b] * p.c[b]) : 0));
    }
    p.W = W;
    p.Siso = Mat3();
    Mat3 Sspec;
    for (int b = 0; b < 3; ++b)
    {
        const Vec3 n(p.N(0, b), p.N(1, b), p.N(2, b));
        p.s[b] = svol[b] + m.g * siso[b];
        p.Siso += outer(n, n) * siso[b];
        Sspec += outer(n, n) * p.s[b];
    }
    p.A = Avol + Aiso * m.g;
    p.S = Sspec + H;
    p.valid = true;
    return p;
}

Mat3 ViscoOgdenFEMForceField::tangent(const PointState& p, const Mat3& dC)
{
    const Mat3 dCp = p.N.transposed() * dC * p.N;   // in the eigenbasis
    Mat3 dSp;
    const Real cmax = std::max({p.c[0], p.c[1], p.c[2]});
    for (int b = 0; b < 3; ++b)
    {
        for (int d = 0; d < 3; ++d)
            dSp(b, b) += p.A(b, d) * dCp(d, d);
        for (int d = 0; d < 3; ++d)
        {
            if (d == b)
                continue;
            Real G;
            const Real gap = p.c[b] - p.c[d];
            if (std::abs(gap) > 1e-6 * cmax)
                G = (p.s[b] - p.s[d]) / gap;
            else   // repeated eigenvalues: limit of the divided difference
                G = 0.5 * ((p.A(b, b) - p.A(b, d)) + (p.A(d, d) - p.A(d, b)));
            dSp(b, d) = G * dCp(b, d);
        }
    }
    return p.N * dSp * p.N.transposed();
}

void ViscoOgdenFEMForceField::elementForces(const Element& e, const PointState& p, std::array<Vec3, 4>& f) const
{
    const Mat3 P = p.F * p.S;
    for (int a = 0; a < 4; ++a)
        f[a] = -(P * e.dN[a]) * e.V0;
}

void ViscoOgdenFEMForceField::elementStiffness(const Element& e, const PointState& p, Mat12& K) const
{
    for (int b = 0; b < 4; ++b)
        for (int k = 0; k < 3; ++k)
        {
            Vec3 ek;
            ek[k] = 1;
            const Mat3 dF = outer(ek, e.dN[b]);
            const Mat3 dC = dF.transposed() * p.F + p.F.transposed() * dF;
            const Mat3 dP = dF * p.S + p.F * tangent(p, dC);
            for (int a = 0; a < 4; ++a)
            {
                const Vec3 df = -(dP * e.dN[a]) * e.V0;
                for (int i = 0; i < 3; ++i)
                    K(3 * a + i, 3 * b + k) = df[i];
            }
        }
}

void ViscoOgdenFEMForceField::addForce(const sofa::core::MechanicalParams*, DataVecDeriv& d_f, const DataVecCoord& d_x,
                                       const DataVecDeriv&)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;
    auto f = sofa::helper::getWriteAccessor(d_f);
    const VecCoord& x = d_x.getValue();
    const Material m = material();
    int inverted = 0;
    for (auto& e : m_elements)
    {
        if (e.V0 <= 0)
            continue;
        const PointState p = evaluate(m, deformationGradient(e, x), e.H);
        e.valid = p.valid;
        if (!p.valid)
        {
            e.K.clear();
            ++inverted;
            continue;
        }
        std::array<Vec3, 4> fe;
        elementForces(e, p, fe);
        for (int a = 0; a < 4; ++a)
            f[e.idx[a]] += fe[a];
        elementStiffness(e, p, e.K);
    }
    if (inverted && m_invertedWarnings < 10)
    {
        ++m_invertedWarnings;
        msg_warning() << inverted << " inverted/degenerate tetrahedra skipped (J <= 0).";
    }
    if (d_checkTangent.getValue() && !m_tangentChecked)
    {
        m_tangentChecked = true;
        checkTangent(x);
    }
}

void ViscoOgdenFEMForceField::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_df,
                                        const DataVecDeriv& d_dx)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;
    auto df = sofa::helper::getWriteAccessor(d_df);
    const VecDeriv& dx = d_dx.getValue();
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
        mparams, this->rayleighStiffness.getValue());
    for (const auto& e : m_elements)
    {
        if (!e.valid)
            continue;
        for (int a = 0; a < 4; ++a)
        {
            Vec3 acc;
            for (int b = 0; b < 4; ++b)
                for (int i = 0; i < 3; ++i)
                    for (int k = 0; k < 3; ++k)
                        acc[i] += e.K(3 * a + i, 3 * b + k) * dx[e.idx[b]][k];
            df[e.idx[a]] += acc * kFactor;
        }
    }
}

void ViscoOgdenFEMForceField::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;
    auto dfdx = matrix->getForceDerivativeIn(this->mstate).withRespectToPositionsIn(this->mstate);
    for (const auto& e : m_elements)
    {
        if (!e.valid)
            continue;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                for (int i = 0; i < 3; ++i)
                    for (int k = 0; k < 3; ++k)
                        dfdx(3 * e.idx[a] + i, 3 * e.idx[b] + k) += e.K(3 * a + i, 3 * b + k);
    }
}

SReal ViscoOgdenFEMForceField::getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord& d_x) const
{
    const VecCoord& x = d_x.getValue();
    Material m = material();
    m.g = 1;
    SReal E = 0;
    for (const auto& e : m_elements)
    {
        if (e.V0 <= 0)
            continue;
        const PointState p = evaluate(m, deformationGradient(e, x), Mat3());
        if (p.valid)
            E += e.V0 * p.W;
    }
    return E;
}

void ViscoOgdenFEMForceField::computeHistory(Element& e, const Mat3& Siso, Real dt) const
{
    const auto& beta = d_beta.getValue();
    const auto& tau = d_tau.getValue();
    e.H = Mat3();
    for (size_t i = 0; i < beta.size(); ++i)
    {
        const Real ef = std::exp(-dt / tau[i]), eh = std::exp(-dt / (2 * tau[i]));
        // Q_i^{n+1} = ef Q_i^n + beta eh (Siso^{n+1} - Siso^n)
        e.Q[i] = e.Q[i] * ef + (Siso - e.SisoPrev) * (beta[i] * eh);
        // history for the next step: Q_i^{n+2} = H_i + beta eh Siso^{n+2}
        e.H += e.Q[i] * ef - Siso * (beta[i] * eh);
    }
    e.SisoPrev = Siso;
}

void ViscoOgdenFEMForceField::commit(const VecCoord& x)
{
    const Real dt = this->getContext()->getDt();
    const bool viscous = d_viscous.getValue() && !d_beta.getValue().empty();
    Material m = material();
    m.g = 1;
    auto qn = sofa::helper::getWriteOnlyAccessor(d_internalStressNorm);
    auto st = sofa::helper::getWriteOnlyAccessor(d_stress);
    qn.resize(m_elements.size());
    st.resize(m_elements.size());
    Real E = 0;
    for (size_t t = 0; t < m_elements.size(); ++t)
    {
        Element& e = m_elements[t];
        if (e.V0 <= 0)
            continue;
        const PointState p = evaluate(m, deformationGradient(e, x), Mat3());
        if (!p.valid)
            continue;   // keep the history of an inverted element unchanged
        E += e.V0 * p.W;
        Mat3 Qsum;
        if (viscous)
        {
            computeHistory(e, p.Siso, dt);
            for (const auto& Q : e.Q)
                Qsum += Q;
        }
        qn[t] = frob(Qsum);
        st[t] = p.S + Qsum;   // S_vol + S_iso + sum Q_i at the committed state
    }
    d_elasticEnergy.setValue(E);
}

void ViscoOgdenFEMForceField::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::AnimateEndEvent::checkEventType(event)
        && this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Valid)
    {
        commit(this->mstate->read(sofa::core::vec_id::read_access::position)->getValue());
    }
}

void ViscoOgdenFEMForceField::checkTangent(const VecCoord& x)
{
    const Material m = material();
    Real maxRel = 0, maxAbs = 0;
    size_t n = 0;
    for (const auto& e : m_elements)
    {
        if (!e.valid || n >= 50)
            continue;
        ++n;
        const Real h = 1e-6 * std::cbrt(e.V0);
        std::array<Coord, 4> xe;
        for (int a = 0; a < 4; ++a)
            xe[a] = x[e.idx[a]];
        auto forces = [&](const std::array<Coord, 4>& y, std::array<Vec3, 4>& f) {
            Mat3 F;
            for (int a = 0; a < 4; ++a)
                F += outer(y[a], e.dN[a]);
            const PointState p = evaluate(m, F, e.H);
            elementForces(e, p, f);
        };
        Real knorm = 0;
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c)
                knorm = std::max(knorm, std::abs(e.K(r, c)));
        for (int b = 0; b < 4; ++b)
            for (int k = 0; k < 3; ++k)
            {
                auto yp = xe, ym = xe;
                yp[b][k] += h;
                ym[b][k] -= h;
                std::array<Vec3, 4> fp, fm;
                forces(yp, fp);
                forces(ym, fm);
                for (int a = 0; a < 4; ++a)
                    for (int i = 0; i < 3; ++i)
                    {
                        const Real fd = (fp[a][i] - fm[a][i]) / (2 * h);
                        const Real err = std::abs(fd - e.K(3 * a + i, 3 * b + k));
                        maxAbs = std::max(maxAbs, err);
                        maxRel = std::max(maxRel, err / std::max(knorm, Real(1e-30)));
                    }
            }
    }
    msg_info() << "checkTangent: " << n << " elements, max |K - K_fd| = " << maxAbs
               << ", relative to max |K| = " << maxRel;
}

void registerViscoOgdenFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData(
                                 "Finite-strain viscoelastic Ogden material (Prony series, internal variables) on linear tetrahedra.")
                                 .add<ViscoOgdenFEMForceField>());
}

} // namespace viscoogden
