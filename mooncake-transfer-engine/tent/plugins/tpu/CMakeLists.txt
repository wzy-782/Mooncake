set(TENT_TPU_ADAPTER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu_pjrt_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu_synthetic_address_registry.cpp
)

add_library(tent_tpu_pjrt_adapter SHARED ${TENT_TPU_ADAPTER_SOURCES})

target_link_libraries(tent_tpu_pjrt_adapter
    PRIVATE tent_plugin ${CMAKE_DL_LIBS}
)

set_target_properties(tent_tpu_pjrt_adapter PROPERTIES
    OUTPUT_NAME "mooncake_tpu_pjrt"
)

install(TARGETS tent_tpu_pjrt_adapter
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)