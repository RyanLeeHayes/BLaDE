/**
 * Standalone BLaDE logging implementation.
 *
 * This definition is only emitted for standalone BLaDE builds.
 * CHARMM provides its own implementation via blade_api.
 */

#include <cstdio>

#ifdef BLADE_STANDALONE
extern "C" void blade_log(const char* message) {
    if (message) {
        printf("%s", message);
        fflush(stdout);
    }
}
#endif
