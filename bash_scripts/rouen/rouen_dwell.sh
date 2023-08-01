python src/train_model.py +data.site_name=ROUEN_FILTERED \
                          +data.time_kind=dwell_times \
                          +model=node_encoded_gcn_1l \
                          ++model.model_parameters.input_size=19 \
                          ++data.rf_remove_zero_obs=True \
                          ++model.model_parameters.empty_graph=False \
                          ++logging_parameters.project_name=ROUEN_dwell_time_RF_GCN
