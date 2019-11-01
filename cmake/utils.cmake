# Get version form include/siech_vec3/version.hpp and put it into SIECH_VEC3_VERSION
function(siech_vec3_extract_version)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/include/vec3/version.hpp" file_contents)
    string(REGEX MATCH "SIECH_VEC3_VERSION_MAJOR ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract major version number from vec3/version.hpp")
    endif()
    set(version_major ${CMAKE_MATCH_1})

    string(REGEX MATCH "SIECH_VEC3_VERSION_MINOR ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract minor version number from vec3/version.hpp")
    endif()
    set(version_minor ${CMAKE_MATCH_1})

    string(REGEX MATCH "SIECH_VEC3_VERSION_PATCH ([0-9]+)" _ "${file_contents}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "Could not extract patch version number from vec3/version.hpp")
    endif()
    set(version_patch ${CMAKE_MATCH_1})

    set(SIECH_VEC3_VERSION_MAJOR ${version_major})
    set(SIECH_VEC3_VERSION "${version_major}.${version_minor}.${version_patch}" PARENT_SCOPE)
endfunction()

# Turn on warnings on the given target
function(siech_vec3_enable_warnings target_name)
    target_compile_options(${target_name} PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>: -Wall -Wextra -Wconversion -pedantic -Wfatal-errors>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>)
endfunction()