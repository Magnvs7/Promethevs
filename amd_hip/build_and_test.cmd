@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

set "ROOT=C:\Users\6700XT\.gemini\antigravity\scratch\Promethevs"
set "ENDO=C:\Users\6700XT\.gemini\antigravity\scratch\endo"
set "AMD_HIP=%ROOT%\amd_hip"
set "BIN=%ROOT%\release_0.2\extracted\PROMETHEVS_0.2\bin"
set "PY=C:\Users\6700XT\AppData\Local\Programs\Python\Python313\python.exe"

cd /d "%AMD_HIP%"
if exist test_hv41c_bitwise.exe del /f /q test_hv41c_bitwise.exe
if exist test_hv41c_bitwise.obj del /f /q test_hv41c_bitwise.obj
if exist test_hv41c_bitwise_runner.dll del /f /q test_hv41c_bitwise_runner.dll
if exist test_hv41c_bitwise_runner.lib del /f /q test_hv41c_bitwise_runner.lib
if exist test_hv41c_bitwise_runner.exp del /f /q test_hv41c_bitwise_runner.exp
if exist test_hv41c_bitwise_runner.obj del /f /q test_hv41c_bitwise_runner.obj

echo === Building hv41cuda.dll (AMD gfx1031 HIP - Zero-Shuffle 32-cols/wave32) ===
cl /nologo /O2 /std:c++17 /utf-8 /EHsc /MD /LD hv41cuda_hip.cpp /Fe:hv41cuda.dll
if errorlevel 1 exit /b 1

echo === Running pure Python+NumPy bitwise and benchmark suite ===
uv run --with numpy --no-project --python "%PY%" python "%AMD_HIP%\test_hv41c_bitwise.py" "%AMD_HIP%\hv41cuda.dll"
exit /b %errorlevel%
