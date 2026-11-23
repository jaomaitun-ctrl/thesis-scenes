#pragma once
/*
 * ViscoOgdenFEMForceField: finite-strain viscoelastic Ogden material on linear tetrahedra
 * (one Gauss point, total Lagrangian).
 *
 * Strain energy (decoupled):  W = U(J) + W_iso(C_bar) + viscous part
 *   W_iso = sum_p 2 mu_p / alpha_p^2 (lb_1^alpha_p + lb_2^alpha_p + lb_3^alpha_p - 3)
 *           lb_a = J^(-1/3) lambda_a (isochoric principal stretches)
 *           -> small-strain shear modulus = sum_p mu_p (Budday et al. 2017 convention)
 *   U     = kappa/2 (ln J)^2
 *
 * Viscoelasticity (Simo 1987, Holzapfel 2000, ch. 6.10): N Prony terms on the isochoric
 * second Piola-Kirchhoff stress, with internal (non-equilibrium) stresses Q_i:
 *   S   = S_vol + S_iso + sum_i Q_i
 *   Q_i^{n+1} = exp(-dt/tau_i) Q_i^n + beta_i exp(-dt/(2 tau_i)) (S_iso^{n+1} - S_iso^n)
 * mu_p are the long-term (relaxed) moduli; the instantaneous ones are (1 + sum beta_i) mu_p.
 * The Q_i are the internal relaxation variables: they carry the loading history and are
 * committed once per time step (AnimateEndEvent), never inside a Newton/LCP iteration.
 * They are exposed per tetrahedron (internalStressNorm, stress) for observers/RL.
 *
 * Tangent: consistent algorithmic tangent of the spectral form
 *   dS/dC = d(S_vol + g S_iso)/dC,  g = 1 + sum_i beta_i exp(-dt/(2 tau_i)),
 * with the limit for repeated eigenvalues. Element stiffness (12x12) is built in addForce
 * and reused by addDForce / buildStiffnessMatrix.
 */
#include <ViscoOgden/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>

#include <array>

namespace viscoogden
{

class SOFA_VISCOOGDEN_API ViscoOgdenFEMForceField
    : public sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>
{
public:
    using DataTypes = sofa::defaulttype::Vec3Types;
    SOFA_CLASS(ViscoOgdenFEMForceField, SOFA_TEMPLATE(sofa::core::behavior::ForceField, DataTypes));

    using Real = SReal;
    using Coord = DataTypes::Coord;
    using Deriv = DataTypes::Deriv;
    using VecCoord = DataTypes::VecCoord;
    using VecDeriv = DataTypes::VecDeriv;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using Vec3 = sofa::type::Vec<3, Real>;
    using Mat3 = sofa::type::Mat<3, 3, Real>;
    using Mat12 = sofa::type::Mat<12, 12, Real>;
    using VecReal = sofa::type::vector<Real>;

    // material
    sofa::Data<VecReal> d_mu;            ///< long-term Ogden moduli mu_p [Pa]
    sofa::Data<VecReal> d_alpha;         ///< Ogden exponents alpha_p
    sofa::Data<Real> d_bulkModulus;      ///< kappa [Pa], U = kappa/2 (ln J)^2
    sofa::Data<VecReal> d_beta;          ///< Prony strain-energy factors beta_i (>= 0)
    sofa::Data<VecReal> d_tau;           ///< Prony relaxation times tau_i [s]
    // options / diagnostics
    sofa::Data<bool> d_checkTangent;     ///< compare the analytic element stiffness with finite differences once
    sofa::Data<bool> d_viscous;          ///< false: freeze history (purely hyperelastic, instantaneous = long-term)
    // outputs (refreshed at each committed step)
    sofa::Data<VecReal> d_internalStressNorm;                ///< |sum_i Q_i| (Frobenius) per tetrahedron [Pa]
    sofa::Data<sofa::type::vector<Mat3>> d_stress;           ///< total 2nd Piola-Kirchhoff stress per tetrahedron [Pa]
    sofa::Data<Real> d_elasticEnergy;                        ///< long-term strain energy (whole mesh) [J]

    sofa::SingleLink<ViscoOgdenFEMForceField, sofa::core::topology::BaseMeshTopology,
                     sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;
    void reinit() override;
    void reset() override;
    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x,
                  const DataVecDeriv& v) override;
    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(sofa::core::behavior::DampingMatrix*) override {}
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord& x) const override;

    /// Constitutive evaluation at one Gauss point (public for tests).
    struct Material
    {
        VecReal mu, alpha;
        Real kappa{0};
        Real g{1};                 ///< scale of the isochoric stress in the current step (1 + sum beta e^{-dt/2tau})
    };
    struct PointState
    {
        bool valid{false};
        Real J{1};
        Mat3 F, S, Siso;           ///< S = S_vol + g S_iso + sum H_i ;  Siso = long-term isochoric stress
        Mat3 N;                    ///< eigenvectors of C (columns)
        Vec3 c;                    ///< eigenvalues of C
        Mat3 A;                    ///< d s_b / d c_d of the spectral part (S_vol + g S_iso)
        Vec3 s;                    ///< spectral values of S_vol + g S_iso
        Real W{0};                 ///< long-term strain energy density
    };
    static PointState evaluate(const Material& m, const Mat3& F, const Mat3& H);
    /// dS for a given dC (consistent tangent of the spectral part; history is constant)
    static Mat3 tangent(const PointState& p, const Mat3& dC);

protected:
    ViscoOgdenFEMForceField();

    struct Element
    {
        std::array<sofa::Index, 4> idx{};
        std::array<Vec3, 4> dN{};  ///< shape-function gradients in the rest configuration
        Real V0{0};
        std::vector<Mat3> Q;       ///< committed internal stresses (one per Prony term)
        Mat3 SisoPrev;             ///< committed long-term isochoric stress
        Mat3 H;                    ///< sum_i of history terms for the coming step
        Mat12 K;                   ///< element stiffness df/dx at the last addForce
        bool valid{false};
    };

    sofa::core::topology::BaseMeshTopology* m_topology{nullptr};
    std::vector<Element> m_elements;
    Real m_dt{0};
    bool m_tangentChecked{false};
    int m_invertedWarnings{0};

    Material material() const;
    Mat3 deformationGradient(const Element& e, const VecCoord& x) const;
    void elementForces(const Element& e, const PointState& p, std::array<Vec3, 4>& f) const;
    void elementStiffness(const Element& e, const PointState& p, Mat12& K) const;
    void commit(const VecCoord& x);
    void computeHistory(Element& e, const Mat3& Siso, Real dt) const;
    void checkTangent(const VecCoord& x);
};

} // namespace viscoogden
