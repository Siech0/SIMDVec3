#############################################################################
# IDE support for headers.
#############################################################################
set(SIECH_VEC3_HEADERS_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

file(GLOB SIECH_VEC3_TOP_HEADERS "${SIECH_VEC3_HEADERS_DIR}/vec3/*.hpp")

set(SIECH_VEC3_ALL_HEADERS ${SIECH_VEC3_TOP_HEADERS})

source_group("Header Files\\siech-vec3" FILES ${SIECH_VEC3_ALL_HEADERS})