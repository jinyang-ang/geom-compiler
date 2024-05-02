#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>  // Include the filesystem header
namespace fs = std::filesystem;

namespace HM5
{
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
        glm::vec3 tangent;
        glm::vec3 bitangent;
    };

    struct QuantizeVertex {
        unsigned int position;
        unsigned int normal;
        glm::vec2 texCoords;
        unsigned int tangent;
        unsigned int bitangent;
    };

    struct Mesh
    {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        uint32_t MaterialGUID = 0;
    };

    struct QuantizeMesh
    {
        std::vector<QuantizeVertex> vertices;
        std::vector<unsigned int> indices;
    };


    struct descriptor
    {
        std::string path_to_intermediate,path_to_texture,mesh_name;
        glm::vec3 scale, rotate, translate;
        void TraverseDirectory(const fs::path& directoryPath);

        enum BV {
            AABB = 0,
            PCA = 1,
            OOBB = 2
        };

        struct Sphere {
            Sphere() = default;
            glm::vec3 min;
            glm::vec3 max;
            glm::vec3 center;
            float radius;
        };

        struct AABBC {
        public:
            AABBC() = default;
            glm::vec3 min;
            glm::vec3 max;

            //OBB
            glm::vec3 center;
            glm::vec3 halfExtents;
            glm::vec3 PA, Y, Z;
        };

        struct PointModel
        {
            // x,y,z
            glm::vec3 position;
            int vertexCount;
        };

        struct BoundingSphereModel
        {
            glm::vec3 position;
            float radius;
        };

        struct parser
        {
            std::string target, desc_path, rsc_path, help;
        };
        parser p_object;
        std::vector<Mesh> mesh;

        void load();
        void LoadModel(std::string filepath);
        void ProcessMesh(aiMesh* aiMesh, const aiScene* scene, std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, uint32_t& matGUID);
        void ProcessNode(aiNode* node, const aiScene* scene, std::vector<Mesh>& meshes);
        //std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);
        void CalculateTangentBitangent(Vertex& vertex, aiVector3D tangent, aiVector3D bitangent);
        void output(std::string filepath);

        // BV
        bool pointSphereIntersection(const PointModel& point, const BoundingSphereModel& sphere);
        std::vector<glm::vec3> verticesBV;
        void LoadVolumes(BV volume, std::vector<glm::vec3> v);
        void GenerateAABB(std::vector<glm::vec3> v);
        void GeneratePCASphere(std::vector<glm::vec3> v);
        void GenerateOBB(std::vector<glm::vec3> v);


        Sphere spherePCA;

        AABBC aabbbox;
        AABBC OBB;
    };
    //function to load descriptor file
}