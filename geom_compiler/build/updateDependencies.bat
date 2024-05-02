@echo off
rmdir "../dependencies/Assimp" /S /Q

REM Assimp
git clone https://github.com/assimp/assimp.git "../dependencies/Assimp"
cd "../dependencies/Assimp"

REM Running cmake to generate the build files and build the dll
cmake -G "Visual Studio 16 2019" -DBUILD_SHARED_LIBS=OFF -DCMAKE_CONFIGURATION_TYPES=Release -DASSIMP_BUILD_TESTS=OFF -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DASSIMP_BUILD_SAMPLES=OFF -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF -DASSIMP_BUILD_FBX_IMPORTER=ON -DASSIMP_BUILD_OBJ_IMPORTER=ON -DASSIMP_BUILD_COLLADA_IMPORTER=ON -DASSIMP_BUILD_GLTF_IMPORTER=ON -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_ZLIB_RECURSIVE=ON -DASSIMP_NO_EXPORT=ON
cmake --build . --config Release

REM Json
rmdir "../../dependencies/Json" /S /Q
git clone https://github.com/nlohmann/json.git "../Json"

REM meshoptimizer
rmdir "../../dependencies/meshoptimizer" /S /Q
git clone https://github.com/zeux/meshoptimizer "../meshoptimizer"
cd "../../dependencies/meshoptimizer"

REM Running cmake to generate the build files and build the library
cmake -DMESHOPT_BUILD_EXAMPLES=OFF -DMESHOPT_BUILD_TESTS=OFF -DMESHOPT_BUILD_SHARED=OFF -B. -H.
cmake --build . --config Release


echo.
echo successfully!
pause
