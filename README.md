Geometry compiler that is used to read in obj and ftx mesh files and convert them into a customized format used in development of custom game engine.
Object's bounding volumes are also computed with the use of quickhull algorithm such as:
1. Axis aligned bounding box (AABB)
2. Object bounding box (OBB)
3. PCA sphere (PCA)

All this information is then compressed as binary and exported as a custom format.
The converted mesh files have a GUID attached to the front for usage within the game engine.

