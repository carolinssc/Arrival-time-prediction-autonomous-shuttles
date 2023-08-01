python src/train_model.py +data.site_name=TAMPERE_FINAL \
                          +data.time_kind=travel_times \
                          +model=node_encoded_gcn_1l \
                          ++model.model_parameters.input_size=18 \
                          ++data.rf_remove_zero_obs=False \
                          ++model.model_parameters.empty_graph=False \
                          ++logging_parameters.project_name=TAMPERE_travel_time_RF_GCN
