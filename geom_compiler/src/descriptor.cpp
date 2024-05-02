
#include "descriptor.h"
#include <iostream>
#include <Eigen/Dense>
#include <QuickHull.hpp>
#include <meshoptimizer.h>

namespace HM5
{
    void descriptor::load()
    {
        //// Read the JSON file
        //std::ifstream file(this->p_object.desc_path);
        //if (!file.is_open()) {
        //    std::cerr << "Error: Failed to open file " << this->p_object.desc_path << std::endl;
        //    return;
        //}
        //json j;
        //file >> j;
        //// Process each object in the JSON array
        //for (const auto& config : j) {
        //    this->path_to_intermediate = config["path_to_intermediate_file"];

        //    // Extract and convert scale, rotation, and translation
        //    std::vector<float> scale = config["scale"];
        //    this->scale = glm::vec3(scale[0], scale[1], scale[2]);

        //    std::vector<float> rotation = config["rotation"];
        //    this->rotate = glm::vec3(rotation[0], rotation[1], rotation[2]);

        //    std::vector<float> translation = config["translation"];
        //    this->translate = glm::vec3(translation[0], translation[1], translation[2]);

        //    this->mesh_name = config["mesh_name"];

        //    // Display the extracted values for each object
        //    std::cout << "Intermediate File Path: " << this->path_to_intermediate << std::endl;
        //    std::cout << "Scale: [" << scale[0] << ", " << scale[1] << ", " << scale[2] << "]" << std::endl;
        //    std::cout << "Rotation: [" << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << "]" << std::endl;
        //    std::cout << "Translation: [" << translation[0] << ", " << translation[1] << ", " << translation[2] << "]" << std::endl;
        //    std::cout << "Mesh Name: " << this->mesh_name << std::endl;

        //   

        //    // call assimp to load
        //    this->LoadModel(*this);

        //    // Reset other member variables for the next object
        //    this->scale = glm::vec3(1.0f);  // Reset to default values or any appropriate defaults
        //    this->rotate = glm::vec3(0.0f);
        //    this->translate = glm::vec3(0.0f);
        //}


        auto GeomPath = fs::current_path();
        TraverseDirectory(GeomPath);
        
    }

    void descriptor::TraverseDirectory(const fs::path& directoryPath) {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            const fs::path& entryPath = entry.path();

            if (fs::is_directory(entryPath)) {
                // Recursively traverse subdirectories
                TraverseDirectory(entryPath);
            }
            else if (fs::is_regular_file(entryPath)) {
                std::cout << "ENTER: " << entryPath.string() << "\n";
                LoadModel(entryPath.string());
            }
        }
    }

    void descriptor::LoadModel(std::string filepath)
    {
        size_t lastdotPos = filepath.find_last_of('.');
        std::string dotobj = filepath.substr(lastdotPos + 1);

        if (dotobj != "obj" && dotobj != "fbx")
            return;

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(filepath
            , aiProcess_Triangulate                // Make sure we get triangles rather than nvert polygons
            | aiProcess_LimitBoneWeights           // 4 weights for skin model max
            | aiProcess_GenUVCoords                // Convert any type of mapping to uv mapping
            //| aiProcess_FindInstances              // search for instanced meshes and remove them by references to one master
            | aiProcess_CalcTangentSpace           // calculate tangents and bitangents if possible
            | aiProcess_JoinIdenticalVertices      // join identical vertices/ optimize indexing
            | aiProcess_RemoveRedundantMaterials   // remove redundant materials
            | aiProcess_FindInvalidData            // detect invalid model data, such as invalid normal vectors
            //| aiProcess_PreTransformVertices       // pre-transform all vertices
            | aiProcess_FlipUVs                    // flip the V to match the Vulkans way of doing UVs
            | aiProcess_GenSmoothNormals           // ensure it has smooth normals 
        );
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cout << "Error loading model: " << importer.GetErrorString() << std::endl;
            return;
        }
        ProcessNode(scene->mRootNode, scene, mesh);

       

        output(filepath);

        //reset
        aabbbox = AABBC();
        OBB = AABBC();
        spherePCA = Sphere();
        verticesBV.clear();
        
    }

    void descriptor::ProcessNode(aiNode* node, const aiScene* scene, std::vector<Mesh>& meshes) {
        // Process all the node's meshes (if any)
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* aiMesh = scene->mMeshes[node->mMeshes[i]];
            std::vector<Vertex> vertices;
            std::vector<unsigned int> indices;
            uint32_t matGUID;
            ProcessMesh(aiMesh, scene, vertices, indices, matGUID);
            meshes.push_back(Mesh(vertices, indices, matGUID));
        }

        // Process all the node's children (if any)
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            ProcessNode(node->mChildren[i], scene, meshes);
        }
    }

    void descriptor::ProcessMesh(aiMesh* aiMesh, const aiScene* scene, std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, uint32_t& matGUID) {

        // Process vertices
        for (unsigned int i = 0; i < aiMesh->mNumVertices; ++i) {
            Vertex vertex;
            vertex.position = glm::vec3(aiMesh->mVertices[i].x, aiMesh->mVertices[i].y, aiMesh->mVertices[i].z);
            vertex.normal = glm::vec3(aiMesh->mNormals[i].x, aiMesh->mNormals[i].y, aiMesh->mNormals[i].z);
            verticesBV.push_back(vertex.position);
            if (aiMesh->mTextureCoords[0]) {
                vertex.texCoords = glm::vec2(aiMesh->mTextureCoords[0][i].x, aiMesh->mTextureCoords[0][i].y);
                CalculateTangentBitangent(vertex, aiMesh->mTangents[i], aiMesh->mBitangents[i]);
            }
            else {
                vertex.texCoords = glm::vec2(0.0f);
            }

            // Process Tangent and BiTangent
            vertices.push_back(vertex);
        }

        // Process indices
        for (unsigned int i = 0; i < aiMesh->mNumFaces; ++i) {
            aiFace face = aiMesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                indices.push_back(face.mIndices[j]);
            }
        }
       
        // Access material index for the mesh
        unsigned int materialIndex = aiMesh->mMaterialIndex;

        // Access the corresponding material
        aiMaterial* material = scene->mMaterials[materialIndex];

        // Print material name
        aiString materialName;
        if (AI_SUCCESS == material->Get(AI_MATKEY_NAME, materialName)) {
            std::cout << "Material Name: " << materialName.C_Str() << std::endl;
            std::hash<std::string> hashFunc;
            size_t hashValue = hashFunc(materialName.C_Str());
            matGUID = static_cast<uint32_t>(hashValue % std::numeric_limits<uint32_t>::max());

        }

    }

    // Function to calculate tangent and bitangent vectors.
    void descriptor::CalculateTangentBitangent(Vertex& vertex, aiVector3D tangent, aiVector3D bitangent) {
        glm::vec3 tangentVec(tangent.x, tangent.y, tangent.z);
        glm::vec3 bitangentVec(bitangent.x, bitangent.y, bitangent.z);

        // Ensure tangent and bitangent vectors are orthogonal to the normal vector.
        glm::vec3 normal = vertex.normal;
        tangentVec = glm::normalize(tangentVec - glm::dot(normal, tangentVec) * normal);
        bitangentVec = glm::normalize(bitangentVec - glm::dot(normal, bitangentVec) * normal);

        vertex.tangent = tangentVec;
        vertex.bitangent = bitangentVec;
    }

    
    void descriptor::LoadVolumes(BV volume, std::vector<glm::vec3> v)
    {

        // write function to generate each BV and store the mesh 
        switch (volume)
        {
        case AABB:
            GenerateAABB(v);
            break;
        case PCA:
            GeneratePCASphere(v);
            break;
        case OOBB:
            GenerateOBB(v);
            break;
        default:
            break;
        }
    }

    void descriptor::GenerateAABB(std::vector<glm::vec3> v)
    {
        // set both to first vertex
        this->aabbbox.min = this->aabbbox.max = v[0];
        for (auto& vert : v)
        {
            // Update minPoint and maxPoint along the X, Y, and Z axes
            this->aabbbox.min.x = std::min(this->aabbbox.min.x, vert.x);
            this->aabbbox.min.y = std::min(this->aabbbox.min.y, vert.y);
            this->aabbbox.min.z = std::min(this->aabbbox.min.z, vert.z);

            this->aabbbox.max.x = std::max(this->aabbbox.max.x, vert.x);
            this->aabbbox.max.y = std::max(this->aabbbox.max.y, vert.y);
            this->aabbbox.max.z = std::max(this->aabbbox.max.z, vert.z);
        }

        float halfExtentX = 0.5f * (this->aabbbox.max.x - this->aabbbox.min.x);
        float halfExtentY = 0.5f * (this->aabbbox.max.y - this->aabbbox.min.y);
        float halfExtentZ = 0.5f * (this->aabbbox.max.z - this->aabbbox.min.z);

        this->aabbbox.halfExtents = glm::vec3(halfExtentX, halfExtentY, halfExtentZ);
        this->aabbbox.PA = { 1.f,0.f,0.f };
        this->aabbbox.Y = { 0.f,1.f,0.f };
        this->aabbbox.Z = { 0.f,0.f,1.f };

        this->aabbbox.center = (this->aabbbox.max + this->aabbbox.min) / 2.f;
    }

    void descriptor::GeneratePCASphere(std::vector<glm::vec3> v)
    {
        // Create QuickHull object
        quickhull::QuickHull<float> qh;
        std::vector<quickhull::Vector3<float>> points;
        for (const auto& vertex : v) {
        	points.emplace_back(vertex.x, vertex.y, vertex.z);
        }

        auto hull = qh.getConvexHull(points, true, false);
        //const auto& indexBuffer = hull.getIndexBuffer();
        const auto& vertexBuffer = hull.getVertexBuffer();


        // Convert convex hull back to glm::vec3 format
        std::vector<glm::vec3> convexHullVertices;
        convexHullVertices.reserve(vertexBuffer.size());
        for (const auto& point : vertexBuffer)
        {
        	convexHullVertices.push_back(glm::vec3(point.x, point.y, point.z));
        }

        // Step 1: Compute the centroid
        glm::vec3 centroid(0.0f);
        for (const glm::vec3& vertex : convexHullVertices) {
        	centroid += vertex;
        }
        centroid /= static_cast<float>(convexHullVertices.size());

        // Step 2: Compute the covariance matrix
        Eigen::Matrix3f covarianceMatrix = Eigen::Matrix3f::Zero();
        for (const glm::vec3& vertex : convexHullVertices) {
        	glm::vec3 deviation = vertex - centroid;
        	covarianceMatrix(0, 0) += deviation.x * deviation.x;
        	covarianceMatrix(0, 1) += deviation.x * deviation.y;
        	covarianceMatrix(0, 2) += deviation.x * deviation.z;
        	covarianceMatrix(1, 0) += deviation.y * deviation.x;
        	covarianceMatrix(1, 1) += deviation.y * deviation.y;
        	covarianceMatrix(1, 2) += deviation.y * deviation.z;
        	covarianceMatrix(2, 0) += deviation.z * deviation.x;
        	covarianceMatrix(2, 1) += deviation.z * deviation.y;
        	covarianceMatrix(2, 2) += deviation.z * deviation.z;
        }
        covarianceMatrix /= static_cast<float>(convexHullVertices.size());

        // Compute eigen vectors and eigen values
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covarianceMatrix);
        Eigen::Matrix3f eigenVectors = eigenSolver.eigenvectors().real();
        Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
        int maxEigenValueIndex = static_cast<int> (eigenValues.maxCoeff());

        // Compute the remaining axes (y-axis and z-axis)
        int yAxis, zAxis;
        if (maxEigenValueIndex == 0) {
        	yAxis = 1;
        	zAxis = 2;
        }
        else if (maxEigenValueIndex == 1) {
        	yAxis = 0;
        	zAxis = 2;
        }
        else {
            maxEigenValueIndex = 2;
        	yAxis = 0;
        	zAxis = 1;
        }

        // Find the maximum and minimum dot products between points and normals
        glm::vec3 normX = { eigenVectors.col(maxEigenValueIndex).x(),eigenVectors.col(maxEigenValueIndex).y(),eigenVectors.col(maxEigenValueIndex).z() };
        glm::vec3 normY = { eigenVectors.col(yAxis).x(),eigenVectors.col(yAxis).y(),eigenVectors.col(yAxis).z() };
        glm::vec3 normZ = { eigenVectors.col(zAxis).x(),eigenVectors.col(zAxis).y(),eigenVectors.col(zAxis).z() };
        std::vector<glm::vec3> norm;
        norm.push_back(normX);
        norm.push_back(normY);
        norm.push_back(normZ);


        // generate mininum sphere using ritter..
        this->spherePCA.min = this->spherePCA.max = v[0];
        for (auto& vert : v)
        {
        	Eigen::Vector3f point =
        		Eigen::Vector3f(vert.x, vert.y, vert.z);
        	float xProjection = point.dot(eigenVectors.col(maxEigenValueIndex));
        	float yProjection = point.dot(eigenVectors.col(yAxis));
        	float zProjection = point.dot(eigenVectors.col(zAxis));

        	// Update minPoint and maxPoint along the X, Y, and Z axes
        	this->spherePCA.min.x = std::min(this->spherePCA.min.x, xProjection);
        	this->spherePCA.min.y = std::min(this->spherePCA.min.y, yProjection);
        	this->spherePCA.min.z = std::min(this->spherePCA.min.z, zProjection);

        	this->spherePCA.max.x = std::max(this->spherePCA.max.x, xProjection);
        	this->spherePCA.max.y = std::max(this->spherePCA.max.y, yProjection);
        	this->spherePCA.max.z = std::max(this->spherePCA.max.z, zProjection);
        }

        // Calculate squared distances along each axis
        float distx = this->spherePCA.max.x - this->spherePCA.min.x;
        float disty = this->spherePCA.max.y - this->spherePCA.min.y;
        float distz = this->spherePCA.max.z - this->spherePCA.min.z;

        // Calculate radius as half the diameter
        spherePCA.radius = std::max(distx, std::max(disty, distz)) / 2.0f;

        // second pass, check if there are any points outside sphere..

        glm::vec3 C = (this->spherePCA.max + this->spherePCA.min) * 0.5f;
        glm::vec3 x = { eigenVectors.col(maxEigenValueIndex).x(),eigenVectors.col(yAxis).x(),eigenVectors.col(zAxis).x() };
        glm::vec3 y = { eigenVectors.col(maxEigenValueIndex).y(),eigenVectors.col(yAxis).y(),eigenVectors.col(zAxis).y() };
        glm::vec3 z = { eigenVectors.col(maxEigenValueIndex).z(),eigenVectors.col(yAxis).z(),eigenVectors.col(zAxis).z() };

        PointModel point;
        BoundingSphereModel sphere;
        sphere.position = { glm::dot(C,x),glm::dot(C,y) ,glm::dot(C,z) };
        sphere.radius = spherePCA.radius;
        for (auto& vert : v)
        {
        	point.position = { glm::dot(vert,x),glm::dot(vert,y) ,glm::dot(vert,z) };
        	if (!pointSphereIntersection(point, sphere)) // point is outside sphere, expand sphere
        	{
        		// Calculate the vector from the current center to the point
        		glm::vec3 centerToPoint = glm::normalize(point.position - sphere.position);
        		// Find point directly opp..
        		glm::vec3 pPrime = sphere.position - sphere.radius * centerToPoint;
        		sphere.position = (point.position + pPrime) / 2.f;
        		sphere.radius = glm::distance(point.position, sphere.position);
        	}

        }
        spherePCA.center = sphere.position;
        spherePCA.radius = sphere.radius;


    }


    bool descriptor::pointSphereIntersection(const PointModel& point, const BoundingSphereModel& sphere)
    {
        // if (c-p)^2 - r^2 <= 0 meaning point is on/inside the sphere
        // dot itself is distance^2
        return (glm::dot((sphere.position - point.position), (sphere.position - point.position)) - (sphere.radius * sphere.radius))
            < std::numeric_limits<float>::epsilon();
    }

    void descriptor::GenerateOBB(std::vector<glm::vec3> v)
    {

        // Create QuickHull object
        quickhull::QuickHull<float> qh;
        std::vector<quickhull::Vector3<float>> points;
        for (const auto& vertex : v) {
        	points.emplace_back(vertex.x, vertex.y, vertex.z);
        }

        auto hull = qh.getConvexHull(points, true, false);
        //const auto& indexBuffer = hull.getIndexBuffer();
        const auto& vertexBuffer = hull.getVertexBuffer();

        // Convert convex hull back to glm::vec3 format
        std::vector<glm::vec3> convexHullVertices;
        convexHullVertices.reserve(vertexBuffer.size());
        for (const auto& point : vertexBuffer)
        {
        	convexHullVertices.push_back(glm::vec3(point.x, point.y, point.z));
        }

        // Step 1: Compute the centroid
        glm::vec3 centroid(0.0f);
        for (const glm::vec3& vertex : convexHullVertices) {
        	centroid += vertex;
        }
        centroid /= static_cast<float>(convexHullVertices.size());

        // Step 2: Compute the covariance matrix
        Eigen::Matrix3f covarianceMatrix = Eigen::Matrix3f::Zero();
        for (const glm::vec3& vertex : convexHullVertices) {
        	glm::vec3 deviation = vertex - centroid;
        	covarianceMatrix(0, 0) += deviation.x * deviation.x;
        	covarianceMatrix(0, 1) += deviation.x * deviation.y;
        	covarianceMatrix(0, 2) += deviation.x * deviation.z;
        	covarianceMatrix(1, 0) += deviation.y * deviation.x;
        	covarianceMatrix(1, 1) += deviation.y * deviation.y;
        	covarianceMatrix(1, 2) += deviation.y * deviation.z;
        	covarianceMatrix(2, 0) += deviation.z * deviation.x;
        	covarianceMatrix(2, 1) += deviation.z * deviation.y;
        	covarianceMatrix(2, 2) += deviation.z * deviation.z;
        }
        covarianceMatrix /= static_cast<float>(convexHullVertices.size());

        // Compute eigen vectors and eigen values
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covarianceMatrix);
        Eigen::Matrix3f eigenVectors = eigenSolver.eigenvectors().real();
        Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
        int maxEigenValueIndex = static_cast<int> (eigenValues.maxCoeff());


        // Compute the remaining axes (y-axis and z-axis)
        int yAxis, zAxis;
        if (maxEigenValueIndex == 0) {
        	yAxis = 1;
        	zAxis = 2;
        }
        else if (maxEigenValueIndex == 1) {
        	yAxis = 0;
        	zAxis = 2;
        }
        else {
            maxEigenValueIndex = 2;
            yAxis = 0;
            zAxis = 1;
        }

        // Compute the minimum and maximum points along all three axes
        glm::vec3 minPoint(std::numeric_limits<float>::max());
        glm::vec3 maxPoint(std::numeric_limits<float>::lowest());


        for (const glm::vec3& vertex : v) {
        	Eigen::Vector3f point =
        		Eigen::Vector3f(vertex.x, vertex.y, vertex.z);

        	float xProjection = point.dot(eigenVectors.col(maxEigenValueIndex));
        	float yProjection = point.dot(eigenVectors.col(yAxis));
        	float zProjection = point.dot(eigenVectors.col(zAxis));

        	minPoint.x = std::min(minPoint.x, xProjection);
        	minPoint.y = std::min(minPoint.y, yProjection);
        	minPoint.z = std::min(minPoint.z, zProjection);

        	maxPoint.x = std::max(maxPoint.x, xProjection);
        	maxPoint.y = std::max(maxPoint.y, yProjection);
        	maxPoint.z = std::max(maxPoint.z, zProjection);
        }

        OBB.halfExtents = glm::vec3((maxPoint - minPoint) * 0.5f);
        OBB.PA = { eigenVectors.col(maxEigenValueIndex).x(),eigenVectors.col(maxEigenValueIndex).y(),eigenVectors.col(maxEigenValueIndex).z() };
        OBB.Y = { eigenVectors.col(yAxis).x(),eigenVectors.col(yAxis).y(),eigenVectors.col(yAxis).z() };
        OBB.Z = { eigenVectors.col(zAxis).x(),eigenVectors.col(zAxis).y(),eigenVectors.col(zAxis).z() };

        glm::vec3 C = (maxPoint + minPoint) * 0.5f;
        glm::vec3 x = { eigenVectors.col(maxEigenValueIndex).x(),eigenVectors.col(yAxis).x(),eigenVectors.col(zAxis).x() };
        glm::vec3 y = { eigenVectors.col(maxEigenValueIndex).y(),eigenVectors.col(yAxis).y(),eigenVectors.col(zAxis).y() };
        glm::vec3 z = { eigenVectors.col(maxEigenValueIndex).z(),eigenVectors.col(yAxis).z(),eigenVectors.col(zAxis).z() };

        OBB.center = { glm::dot(C,x),glm::dot(C,y) ,glm::dot(C,z) };
    }


    void descriptor::output(std::string filepath)
    {
        size_t lastSlashPos = filepath.find_last_of('\\');
        size_t lastdotPos = filepath.find_last_of('.');
        std::string type = filepath.substr(lastSlashPos + 1, lastdotPos - lastSlashPos - 1);


        // Define the folder path where you want to store files
        std::string folderPath = type;  // Change this to your desired folder name

        // Create the folder if it doesn't exist
        if (!fs::exists(folderPath)) {
            if (!fs::create_directory(folderPath)) {
                std::cerr << "Failed to create the folder." << std::endl;
                return;
            }
        }
        std::hash<std::string> hashFunc;
        size_t hashValue = hashFunc(type);

        // Convert the hashValue to a uint32_t without data loss ensuring within range of uint32_t
        uint32_t GUID = static_cast<uint32_t>(hashValue % std::numeric_limits<uint32_t>::max());


        // Write the optimized mesh to a binary file
        std::ofstream outFile(fs::path(folderPath) / fs::path(std::to_string(GUID) + "_output.mesh" ), std::ios::binary);

        if (!outFile.is_open()) {
            std::cerr << "Failed to open the file for writing." << std::endl;
            return;
        }

        // Write the number of meshes in the file
        size_t numMeshes = mesh.size();
        outFile.write(reinterpret_cast<const char*>(&numMeshes), sizeof(numMeshes));

       for ( Mesh& meshes : mesh) 
       {
           for (int i = AABB; i <= OOBB; i++)
               LoadVolumes(static_cast<BV>(i), verticesBV);

           size_t vertexCount = meshes.vertices.size();
           size_t indexCount = meshes.indices.size();

           //optimise
           std::vector<unsigned int> remap(indexCount); // allocate temporary memory for the remap table
           size_t vertex_count = meshopt_generateVertexRemap(&remap[0], meshes.indices.data(), indexCount, meshes.vertices.data(), vertexCount, sizeof(Vertex));

           Mesh temp;
           temp.indices.resize(indexCount);
           temp.vertices.resize(vertex_count);

           meshopt_remapIndexBuffer(temp.indices.data(), meshes.indices.data(), indexCount, &remap[0]);
           meshopt_remapVertexBuffer(temp.vertices.data(), meshes.vertices.data(), vertex_count, sizeof(Vertex), &remap[0]);
           // Vertex cache optimization
           meshopt_optimizeVertexCache(temp.indices.data(), temp.indices.data(), indexCount, vertex_count);
           //Overdraw optimization
           meshopt_optimizeOverdraw(temp.indices.data(), temp.indices.data(), indexCount, &temp.vertices[0].position.x, vertex_count, sizeof(Vertex), 1.05f);
           //Vertex fetch optimization
           meshopt_optimizeVertexFetch(temp.vertices.data(), temp.indices.data(), indexCount, temp.vertices.data(), vertex_count, sizeof(Vertex));

           vertexCount = temp.vertices.size();
           indexCount = temp.indices.size();

          /* QuantizeMesh Quantizetemp;
           Quantizetemp.vertices.resize(vertexCount);
           for (size_t i = 0; i < temp.vertices.size(); ++i)
           {
               Quantizetemp.vertices[i].position = (meshopt_quantizeUnorm(temp.vertices[i].position.x, 10) << 20) |
                   (meshopt_quantizeUnorm(temp.vertices[i].position.y, 10) << 10) |
                   meshopt_quantizeUnorm(temp.vertices[i].position.z, 10);

               Quantizetemp.vertices[i].normal = (meshopt_quantizeUnorm(temp.vertices[i].normal.x, 10) << 20) |
                   (meshopt_quantizeUnorm(temp.vertices[i].normal.y, 10) << 10) |
                   meshopt_quantizeUnorm(temp.vertices[i].normal.z, 10);

               Quantizetemp.vertices[i].bitangent = (meshopt_quantizeUnorm(temp.vertices[i].bitangent.x, 10) << 20) |
                   (meshopt_quantizeUnorm(temp.vertices[i].bitangent.y, 10) << 10) |
                   meshopt_quantizeUnorm(temp.vertices[i].bitangent.z, 10);

               Quantizetemp.vertices[i].tangent = (meshopt_quantizeUnorm(temp.vertices[i].tangent.x, 10) << 20) |
                   (meshopt_quantizeUnorm(temp.vertices[i].tangent.y, 10) << 10) |
                   meshopt_quantizeUnorm(temp.vertices[i].tangent.z, 10);
           }*/

           // Write the vertex count
           outFile.write(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));

           // Write the vertices
           outFile.write(reinterpret_cast<const char*>(temp.vertices.data()), vertexCount * sizeof(Vertex));

           // Write the index count
           outFile.write(reinterpret_cast<char*>(&indexCount), sizeof(indexCount));

           // Write the indices
           outFile.write(reinterpret_cast<const char*>(temp.indices.data()), indexCount * sizeof(unsigned int));

           // Write the MaterialGUID
           outFile.write(reinterpret_cast<char*>(&meshes.MaterialGUID), sizeof(uint32_t));
       }


       // Write the AABB
       outFile.write(reinterpret_cast<const char*>(&aabbbox),sizeof(AABBC));
       // Write the PCA
       outFile.write(reinterpret_cast<const char*>(&spherePCA), sizeof(Sphere));
       // Write the OBB
       outFile.write(reinterpret_cast<const char*>(&OBB), sizeof(AABBC));


       outFile.close();
       std::cout << "Geom Data written to: " << (fs::path(folderPath) / fs::path(std::to_string(GUID) + "_output.mesh" )).string() << std::endl;

       mesh.clear();

    }
}