@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo ERROR: Failed to initialize VS 2022 Build Tools x64 environment.
    exit /b 1
)

set "ROOT=C:\Users\6700XT\.gemini\antigravity\scratch\Promethevs"
set "AMD_HIP=%ROOT%\amd_hip"
set "FROZEN=%ROOT%\release_0.2\extracted\PROMETHEVS_0.2\source\frozen"
set "BIN=%ROOT%\release_0.2\extracted\PROMETHEVS_0.2\bin"
set "PY=C:\Users\6700XT\AppData\Local\Programs\Python\Python313\python.exe"

cd /d "%AMD_HIP%"

echo === [1/2] Building hv41cuda.dll (AMD gfx1031 HIP) ===
cl /nologo /O2 /arch:AVX2 /std:c++17 /utf-8 /EHsc /MD /LD ^
   "%AMD_HIP%\hv41cuda_hip.cpp" ^
   /Fe:"%AMD_HIP%\hv41cuda.dll"
if errorlevel 1 exit /b 1

echo === [2/2] Linking v41_serve2_BF.exe (AMD PROMETHEVS host server) ===
cl /nologo /O2 /arch:AVX2 /std:c++17 /utf-8 /EHsc /MD /openmp ^
   /I "%AMD_HIP%\include" ^
   /I "%FROZEN%\core" ^
   "%FROZEN%\core\v41_serve2.cpp" ^
   "%FROZEN%\obj\deepseek41.cpp.obj" ^
   "%FROZEN%\libs\llama.lib" ^
   "%FROZEN%\libs\ggml-cpu.lib" ^
   "%FROZEN%\libs\ggml-base.lib" ^
   "%FROZEN%\libs\ggml.lib" ^
   ws2_32.lib advapi32.lib psapi.lib ^
   /Fo:"%AMD_HIP%\v41_serve2.obj" ^
   /Fe:"%AMD_HIP%\v41_serve2_BF.exe"
if errorlevel 1 exit /b 1

echo === Build succeeded: amd_hip\hv41cuda.dll and amd_hip\v41_serve2_BF.exe ===

if /I not "%~1"=="--stage" (
    echo Run with --stage from a regular Windows terminal to stage into PROMETHEVS_0.2\bin\ and smoke-test.
    exit /b 0
)

echo === Staging into %BIN% ===
if not exist "%BIN%\hv41cuda_nvidia_orig.dll" if exist "%BIN%\hv41cuda.dll" (
    copy /Y "%BIN%\hv41cuda.dll" "%BIN%\hv41cuda_nvidia_orig.dll" >nul
)
if not exist "%BIN%\v41_serve2_BF_orig.exe" if exist "%BIN%\v41_serve2_BF.exe" (
    copy /Y "%BIN%\v41_serve2_BF.exe" "%BIN%\v41_serve2_BF_orig.exe" >nul
)
copy /Y "%AMD_HIP%\hv41cuda.dll" "%BIN%\hv41cuda.dll" >nul
if errorlevel 1 exit /b 1
copy /Y "%AMD_HIP%\v41_serve2_BF.exe" "%BIN%\v41_serve2_BF.exe" >nul
if errorlevel 1 exit /b 1

echo === Smoke-testing staged binaries ===
"%BIN%\v41_serve2_BF.exe"
set "USAGE_RC=%errorlevel%"
if not "%USAGE_RC%"=="1" (
    echo ERROR: staged v41_serve2_BF.exe failed with exit %USAGE_RC% -- check Smart App Control.
    exit /b 1
)
if not exist "%PY%" goto :skip_py_smoke
"%PY%" -c "import ctypes, os; os.add_dll_directory(r'%BIN%'); d=ctypes.WinDLL(r'%BIN%\hv41cuda.dll'); b=ctypes.create_string_buffer(256); rc=d.hv41c_init(b,256); print('hv41c_init:', rc, b.value.decode()); assert rc==0"
if errorlevel 1 exit /b 1
:skip_py_smoke
echo === Phase 3 stage and smoke check PASSED ===
exit /b 0
