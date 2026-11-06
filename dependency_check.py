# dependency_check.py
import sys
import subprocess
import importlib.util
import shutil

# Core package mappings: {import_name: pip_package_name}
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "scipy": "scipy",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "netCDF4": "netCDF4",
    "cfgrib": "cfgrib",
    "eccodes": "eccodes",
    "cdsapi": "cdsapi",
}


def check_and_install_dependencies():
    """
    Checks if all required Python packages exist. If missing, attempts automated
    pip install/upgrade and verifies cfgrib/eccodes system bindings.
    """
    print("=" * 70)
    print("  STANDALONE ENVIRONMENT DEPENDENCY & RUNTIME VERIFICATION")
    print("=" * 70)

    missing = []
    for pkg_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(pkg_name) is None:
            print(f"[MISSING] Package '{pkg_name}' is not installed.")
            missing.append(pip_name)
        else:
            print(f"[FOUND]   Package '{pkg_name}' is available.")

    # 1. Attempt automated installation of missing packages
    if missing:
        print(f"\n[INSTALL] Installing missing dependencies: {', '.join(missing)}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", *missing
            ])
            print("[SUCCESS] Missing packages successfully installed.\n")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to pip install packages (Code {e.returncode}).")
            _print_conda_recommendation()
            return False

    # 2. In-Depth cfgrib & ECMWF eccodes Binary Verification
    cfgrib_ok = verify_cfgrib_engine()
    if not cfgrib_ok:
        print("[WARN] GRIB2 files will fall back to synthetic data until eccodes is resolved.")
        return False

    print("[SUCCESS] All environment requirements verified.\n")
    return True


def verify_cfgrib_engine():
    """
    Validates that cfgrib can communicate with libeccodes C-engine.
    """
    try:
        import cfgrib
        # Test basic internal eccodes binding
        import eccodes
        v = eccodes.codes_get_api_version()
        print(f"[VERIFIED] cfgrib + libeccodes engine active (API v{v}).")
        return True
    except Exception as e:
        print(f"[WARN] 'cfgrib' installed but failed binary binding check: {e}")
        # Attempt to upgrade/re-link eccodes python wrapper
        try:
            print("[REPAIR] Attempting pip install --upgrade --force-reinstall eccodes cfgrib...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "eccodes", "cfgrib"
            ])
            import eccodes
            print("[SUCCESS] Successfully repaired eccodes wrapper bindings.")
            return True
        except Exception:
            _print_conda_recommendation()
            return False


def _print_conda_recommendation():
    print("\n" + "*" * 70)
    print("  GRIB2 ECC-CODES BINARY NOTICE")
    print("  On standalone/air-gapped systems, GRIB2 decoding relies on the")
    print("  pre-compiled C-library 'eccodes'. If pip cannot build it, execute:")
    print("    conda install -c conda-forge cfgrib eccodes -y")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    success = check_and_install_dependencies()
    sys.exit(0 if success else 1)
