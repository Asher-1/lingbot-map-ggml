# Apply the one tracked ggml patch at configure time. It must either apply
# cleanly or reverse-apply cleanly; any mixed state is a configuration error.
function(lingbot_apply_ggml_patches GGML_DIR PATCH_DIR)
    file(GLOB PATCHES "${PATCH_DIR}/*.patch")
    list(SORT PATCHES)
    list(LENGTH PATCHES PATCH_COUNT)
    if(NOT PATCH_COUNT EQUAL 1)
        message(FATAL_ERROR "[lingbot ggml patches] expected exactly one consolidated patch, found ${PATCH_COUNT}")
    endif()
    find_program(GIT_EXECUTABLE git REQUIRED)
    foreach(PATCH IN LISTS PATCHES)
        get_filename_component(NAME "${PATCH}" NAME)
        execute_process(COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH}"
            WORKING_DIRECTORY "${GGML_DIR}" RESULT_VARIABLE FORWARD
            OUTPUT_QUIET ERROR_QUIET)
        if(FORWARD EQUAL 0)
            execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${PATCH}"
                WORKING_DIRECTORY "${GGML_DIR}" RESULT_VARIABLE APPLIED
                OUTPUT_QUIET ERROR_VARIABLE ERROR_TEXT)
            if(NOT APPLIED EQUAL 0)
                message(FATAL_ERROR "[lingbot ggml patches] failed applying ${NAME}: ${ERROR_TEXT}")
            endif()
            message(STATUS "[lingbot ggml patches] applied ${NAME}")
        else()
            execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH}"
                WORKING_DIRECTORY "${GGML_DIR}" RESULT_VARIABLE REVERSE
                OUTPUT_QUIET ERROR_QUIET)
            if(REVERSE EQUAL 0)
                message(STATUS "[lingbot ggml patches] already applied ${NAME}")
            else()
                message(FATAL_ERROR "[lingbot ggml patches] ${NAME} conflicts or is partially applied; reset this submodule to v0.21.0")
            endif()
        endif()
    endforeach()
endfunction()
