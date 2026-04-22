#ifndef MAIN_BLADE_LOG_H
#define MAIN_BLADE_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Platform-independent logging function.
 *
 * BLaDE declares this function but does NOT implement it.
 * The implementation must be provided by whoever links BLaDE:
 *
 * - Standalone: blade_log.cpp in main/ (printf to stdout)
 * - CHARMM: blade_api/blade_main.F90 (routes to OUTU)
 * - Other: provide your own implementation
 */
void blade_log(const char* message);

#ifdef __cplusplus
}
#endif

#endif // MAIN_BLADE_LOG_H
