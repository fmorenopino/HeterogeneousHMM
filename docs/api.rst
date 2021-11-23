API Reference
=============

This is the class and function reference of ``heterogeneoushmm``.

Please refer to the :ref:`full user guide <user_guide>` for further details, as
the class and function raw specifications may not be enough to give full
guidelines on their uses.


BaseHMM
~~~~~~~~

.. autoclass:: heterogeneoushmm.base.BaseHMM
   :exclude-members: set_params, get_params, _get_param_names, __init__
   :private-members: False
   :no-inherited-members: True


GaussianHMM
~~~~~~~~~~~

.. autoclass:: heterogeneoushmm.gaussian.GaussianHMM
   :exclude-members: covars, set_params, get_params, __init__
   :private-members: False
   :no-inherited-members: True


MultinomialHMM
~~~~~~~~~~~~~~

.. autoclass:: heterogeneoushmm.multinomial.MultinomialHMM
   :exclude-members: set_params, get_params, __init__
   :private-members: False
   :no-inherited-members: True

HeterogeneousHMM
~~~~~~~~~~~~~~~~

.. autoclass:: heterogeneoushmm.heterogeneous.HeterogeneousHMM
   :exclude-members: covars, set_params, get_params, __init__
   :private-members: False
   :no-inherited-members: True

Utils
~~~~~

.. automodule:: heterogeneoushmm.utils
   :exclude-members: normalise, log_normalise, log_mask_zero, concatenate_observation_sequences, init_covars, fill_covars, validate_covars, check_if_attributes_set, plot_log_likelihood_evolution, plot_predictions, plot_latent_state_sequence, create_emissions_name_list, print_table, print_startprob_table, print_transition_table, print_emission_table, print_emission_table_het, print_means_table, print_covars_table