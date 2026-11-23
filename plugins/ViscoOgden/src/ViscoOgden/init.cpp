#include <ViscoOgden/config.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace viscoogden
{
extern void registerViscoOgdenFEMForceField(sofa::core::ObjectFactory* factory);

extern "C" {
SOFA_VISCOOGDEN_API void initExternalModule();
SOFA_VISCOOGDEN_API const char* getModuleName();
SOFA_VISCOOGDEN_API const char* getModuleVersion();
SOFA_VISCOOGDEN_API const char* getModuleLicense();
SOFA_VISCOOGDEN_API const char* getModuleDescription();
SOFA_VISCOOGDEN_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);
    }
}
const char* getModuleName() { return MODULE_NAME; }
const char* getModuleVersion() { return MODULE_VERSION; }
const char* getModuleLicense() { return "LGPL"; }
const char* getModuleDescription()
{
    return "Finite-strain viscoelastic Ogden material (Prony series on the isochoric stress, "
           "Simo/Holzapfel internal variables) for linear tetrahedra.";
}
void registerObjects(sofa::core::ObjectFactory* factory) { registerViscoOgdenFEMForceField(factory); }
}
