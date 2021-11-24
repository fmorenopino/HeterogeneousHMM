""""""
"""
Created on Nov 20, 2019
@author: semese

Parts of the code come from: https://github.com/hmmlearn/hmmlearn
"""


import pickle
import numpy as np
from scipy import linalg, special
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('seaborn-ticks')

COVARIANCE_TYPES = frozenset(('spherical', 'tied', 'diagonal', 'full'))
COLORS = sns.color_palette('colorblind', n_colors=15)

# ---------------------------------------------------------------------------- #
#                            Utils for the HMM models                          #
# ---------------------------------------------------------------------------- #


def normalise(a, axis=None):
    """
    Normalise the input array so that it sums to 1.  Modifies the input **inplace**.

    :param a: non-normalised input data
    :type a: array_like
    :param axis: dimension along which normalisation is performed, defaults to None
    :type axis: int, optional
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalise(a, axis=None):
    """
    Normalise the input array so that ``sum(exp(a)) == 1``. Modifies the input **inplace**.

    :param a: non-normalised input data
    :type a: array_like
    :param axis: dimension along which normalisation is performed, defaults to None
    :type axis: int, optional
    """
    with np.errstate(under='ignore'):
        a_lse = special.logsumexp(a, axis, keepdims=True)
    a -= a_lse


def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalised to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.

    :param a: input data
    :type a: array_like
    """
    a = np.asarray(a)
    with np.errstate(divide='ignore'):
        return np.log(a)


def concatenate_observation_sequences(observation_sequences, gidx=None):
    """
    Function to concatenate the observation sequences and remove the
    partially or completely missing observations to create a proper
    input for the KMeans.

    :param observation_sequences: each element is an array of observations
    :type observation_sequences: list
    :param gidx: if provided, only the specified columns will be concatenated, 
        defaults to None
    :type gidx: array_like, optional
    :return: concatenated observations without missing values
    :rtype: list
    """
    concatenated = []
    for obs_seq in observation_sequences:
        for obs in obs_seq:
            if gidx is not None:
                gidx = int(gidx)
                if not np.any(obs[:gidx] is np.nan or obs[:gidx] != obs[:gidx]):
                    concatenated.append(obs[:gidx])
            else:
                if not np.any(obs is np.nan or obs != obs):
                    concatenated.append(obs)
    return np.asarray(concatenated, dtype=float)


def init_covars(tied_cv, covariance_type, n_states):
    """
    Method for initialising the covariances based on the
    covariance type. See GaussianHMM class definition for details.

    :param tied_cv: the tied covariance matrix 
    :type tied_cv: array_like
    :param covariance_type: covariance_type: string describing the type of
            covariance parameters to use
    :type covariance_type: string
    :param n_states: number of hidden states 
    :type n_states: int
    :return: the initialised covariance matrix
    :rtype: array_like
    """
    if covariance_type == 'spherical':
        cv = tied_cv.mean() * np.ones((n_states,))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diagonal':
        cv = np.tile(np.diag(tied_cv), (n_states, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_states, 1, 1))
    return cv


def fill_covars(covars, covariance_type, n_states, n_features):
    """
    Return the covariance matrices in full form: (n_states, n_features, n_features)

    :param covars: the reduced form of the covariance matrix
    :type covars: array_like
    :param covariance_type: covariance_type: string describing the type of
            covariance parameters to use
    :type covariance_type: string
    :param n_states: number of hidden states 
    :type n_states: int
    :param n_features: the number of features
    :type n_features: int
    :return: the covariance matrices in full form: (n_states, n_features, n_features)
    :rtype: array_like
    """
    new_covars = np.array(covars, copy=True)
    if covariance_type == 'full':
        return new_covars
    elif covariance_type == 'diagonal':
        return np.array(list(map(np.diag, new_covars)))
    elif covariance_type == 'tied':
        return np.tile(new_covars, (n_states, 1, 1))
    elif covariance_type == 'spherical':
        eye = np.eye(n_features)[np.newaxis, :, :]
        new_covars = new_covars[:, np.newaxis, np.newaxis]
        temp = eye * new_covars
        return temp

# Copied from scikit-learn 0.19.


def validate_covars(covars, covariance_type, n_states):
    """Do basic checks on matrix covariance sizes and values."""

    if covariance_type == 'spherical':
        if len(covars) != n_states:
            raise ValueError('"sperical" covars have length n_states')
        elif np.any(covars <= 0):
            raise ValueError('"sperical" covars must be non-negative')
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError('"tied" covars must have shape (n_dim, n_dim)')
        elif not np.allclose(covars, covars.T) or np.any(linalg.eigvalsh(covars) <= 0):
            raise ValueError(
                '"tied" covars must be symmetric, ' 'positive-definite')
    elif covariance_type == 'diagonal':
        if len(covars.shape) != 2:
            raise ValueError(
                '"diagonal" covars must have shape ' '(n_states, n_dim)')
        elif np.any(covars <= 0):
            raise ValueError('"diagonal" covars must be non-negative')
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError(
                '"full" covars must have shape ' '(n_states, n_dim, n_dim)'
            )
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError(
                '"full" covars must have shape ' '(n_states, n_dim, n_dim)'
            )
        for n, cv in enumerate(covars):
            if not np.allclose(cv, cv.T) or np.any(linalg.eigvalsh(cv) <= 0):
                raise ValueError(
                    'component %d of "full" covars must be '
                    'symmetric, positive-definite' % n
                )
    else:
        raise ValueError(
            'covariance_type must be one of '
            + '"sperical", "tied", "diagonal", "full"'
        )


def check_if_attributes_set(model, attr=None):
    """
    Checks if the model attributes are set before training. This is only necessary
    if the 'no_init' option is selected, because in that case the model parameters
    are expected to be set apriori, and won't be reinitialised before training or
    if some of the discrete emission probabilities aren't trained.

    :param model: an HMM model
    :type model: object
    :param attr: which attributes to check, defaults to None
    :type attr: str, optional
    :raises AttributeError: if one of the tested parameters is not initialised
    """
    params_dict = {'t': 'A', 's': 'pi', 'e': 'B', 'm': 'means', 'c': 'covars'}
    model_dict = model.__dict__
    if attr is not None:
        if not params_dict[attr] in model_dict.keys():
            raise AttributeError(
                'Attr self.'
                + params_dict[attr]
                + ' must be initialised before training'
            )
    else:
        for par in model.tr_params:
            if params_dict[par] in model_dict.keys():
                continue
            else:
                raise AttributeError(
                    'Attr self.'
                    + params_dict[par]
                    + ' must be initialised before training'
                )

# ---------------------------------------------------------------------------- #
#                           Model order selection utils                        #
# ---------------------------------------------------------------------------- #


def aic_hmm(log_likelihood, dof):
    """
    Function to compute the Aikaike's information criterion for an HMM given
    the log-likelihood of observations.

    :param log_likelihood: logarithmised likelihood of the model
        dof (int) - single numeric value representing the number of trainable
        parameters of the model
    :type log_likelihood: float
    :param dof: single numeric value representing the number of trainable
        parameters of the model
    :type dof: int
    :return: the Aikaike's information criterion
    :rtype: float
    """
    return -2 * log_likelihood + 2 * dof


def bic_hmm(log_likelihood, dof, n_samples):
    """
    Function to compute Bayesian information criterion for an HMM given a
    the log-likelihood of observations.

    :param log_likelihood: logarithmised likelihood of the model
        dof (int) - single numeric value representing the number of trainable
        parameters of the model
    :type log_likelihood: float
    :param dof: single numeric value representing the number of trainable
        parameters of the model
    :type dof: int
    :param n_samples: length of the time-series of observations
    :type n_samples: int
    :return: the Bayesian information criterion
    :rtype: float
    """
    return -2 * log_likelihood + dof * np.log(n_samples)


def get_n_fit_scalars(hmm):
    """
    Function to compute the degrees of freedom of a HMM based on the parameters
    that are trained in it.

    :param hmm: a hidden Markov model (GaussianHMM, MultinomialHMM, etc)
    :type hmm: object
    :return: single numeric value representing the number of trainable
        parameters of the model
    :rtype: int

    """
    train_params = hmm.tr_params
    n_fit_scalars_per_param = hmm.get_n_fit_scalars_per_param()
    dof = 0
    for par in n_fit_scalars_per_param:
        if par in train_params:
            dof += n_fit_scalars_per_param[par]

    return dof

# ---------------------------------------------------------------------------- #
#                             Visualisation utils                              #
# ---------------------------------------------------------------------------- #


def plot_log_likelihood_evolution(log_likelihood, filename=None):
    """ 
    Plot the evolution of the log-likelihood over the iterations.

    :param log_likelihood: the list of log-likelihood values
    :type log_likelihood: list
    :param filename: full path to where to save the figure, defaults to None
    :type filename: string, optional
    """
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    iters = np.arange(1, len(log_likelihood) + 1)
    ax.plot(iters, log_likelihood)
    ax.set_xlabel('# iterations')
    ax.set_ylabel('Log-likelihood')

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_model_selection(n_states, criteria, filename=None):
    """
    Function to plot the different model order selection criteria vs the
    number of states of a HMM.

    :param n_states: list of number of states that were used to compute
                the criteria
    :type n_states: list
    :param criteria: the keys correspond to the different criteria
                that were computed (AIC, BIC, etc) and the values are a list of
                length len(n_states) containing the criteria values for the
                different number of states
    :type criteria: dict
    :param filename: full path to where to save the figure, defaults to None
    :type filename: string, optional
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for key, value in criteria.items():
        ax.plot(n_states, value, marker='o', label=key)

    ax.set_frame_on(True)
    ax.set_xlabel('# states')
    ax.set_ylabel('Criterion')
    ax.legend()

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_decode(
    obs_seq,
    data_cols,
    state_seq,
    discrete_columns=None,
    state_names=None,
    time_stamps=None,
    figsize=(10, 20),
    filename=None,
):
    """
    Function for plotting the decoded sequence with the corresponding states.

    :param obs_seq: an observation sequence
    :type obs_seq: array_like
    :param data_cols: name of (Gaussian) data columns (name of each column)
    :type data_cols: list
    :param state_seq: the decoded state sequence
    :type state_seq: list
    :param discrete_columns: only in case of HMM with labels, the
                list of label names, defaults to None
    :type discrete_columns: list, optional
    :param state_names: list of state names, only if applicable, defaults to None
    :type state_names: list, optional
    :param time_stamps: list of time_stamps when the observations where
                recorded (if applicable, otherwise discrete values are applied, starting
                of 0 to len(state_seq)), defaults to None
    :type time_stamps: [list, optional
    :param figsize: size of the plot, defaults to (10, 20)
    :type figsize: tuple, optional
    :param filename: full path with figure name if it should be saved, defaults to None
    :type filename: str, optional
    """
    if len(data_cols) > 1:
        fig, axes = plt.subplots(nrows=len(data_cols), figsize=figsize)
        for i, (ax, variable) in enumerate(zip(axes, data_cols)):
            if discrete_columns is not None and variable in discrete_columns:
                ax.set_ylim(
                    [np.amin(obs_seq[:, i]) - 0.1,
                     np.amax(obs_seq[:, i]) + 0.1]
                )
            plot_predictions(
                obs_seq[:, i],
                variable,
                state_seq,
                ax,
                state_names=state_names,
                time_stamps=time_stamps,
                first=(i == 0),
            )
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        plot_predictions(
            obs_seq,
            data_cols[0],
            state_seq,
            ax,
            time_stamps=time_stamps,
            state_names=state_names,
            first=True,
        )
        fig.tight_layout()

    fig.autofmt_xdate()

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_predictions(
    data, variable, states, ax, time_stamps=None, state_names=None, first=False
):
    """
    Helper function for plotting the decoded sequence with the corresponding
    states.
    """
    # Plot the observed data.
    x_observed = np.arange(len(data))
    y_observed = data
    ax.plot(
        x_observed,
        y_observed,
        label=variable,
        color='k',
        marker='.',
        markersize=5,
        linewidth=1,
    )
    #  Optionally plot latent temporal state at each timepoint,
    #  according to a given chain in the model.
    plot_latent_state_sequence(
        x_observed, y_observed, states, ax, state_names=state_names, first=first
    )
    # if dates are provided, set them as labels
    if time_stamps is not None:
        ax.set_xticklabels([ts.strftime('%m/%d/%Y, %H:%M')
                           for ts in time_stamps])
    # Add the legend.
    # ax.legend(loc='upper right', handletextpad=0)
    ax.set_title(variable)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_latent_state_sequence(
    timesteps, values, states, ax, state_names=None, first=False
):
    """
    Helper function for plotting the decoded sequence with the corresponding
    states.
    """
    unique = sorted(set(states))
    colors = [COLORS[i] for i in unique]
    y_low, y_high = ax.get_ylim()
    y_mid = np.mean([y_low, y_high])
    y_height = 0.05 * (y_high - y_low)
    custom_lines = []
    for state, color in zip(unique, colors):
        xs = timesteps[states == state]
        for x in xs:
            if x > 0:
                ax.fill_between(
                    [x - 1, x],
                    [y_mid - y_height] * 2,
                    [y_mid + y_height] * 2,
                    alpha=0.3,
                    color=color,
                )
        if first:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

    if first:
        if state_names is not None:
            ax.legend(
                custom_lines,
                state_names,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                frameon=True,
            )
        else:
            ax.legend(
                custom_lines,
                ['State_' + str(unq) for unq in unique],
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                frameon=True,
            )


# ---------------------------------------------------------------------------- #
#                                   IO utils                                   #
# ---------------------------------------------------------------------------- #
def save_model(model, filename):
    """
    Function to save an hmm model (or any variable) to a pickle file.

    :param model: the trained HMM to be saved
    :type model: object
    :param filename: full path or just file name where to save the variable
    :type filename: str
    """
    with open(filename, 'wb2') as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Function to load an HMM model from a pickle file.

    :param filename: full path or just file name where to save the variable
    :type filename: str
    :return: the trained HMM that was in the file
    :rtype: object
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def pretty_print_hmm(model, hmm_type='Multinomial', states=None, emissions=None):
    """
    Function to pretty print the parameters of an hmm model.

    :param model: and HMM object
    :type model: object
    :param hmm_type: the type of the HMM model; can be 'Multinomial',
                'Gaussian' or 'Heterogeneous', defaults to 'Multinomial'
    :type hmm_type: str, optional
    :param states: list with the name of states, if any, defaults to None
    :type states: list, optional
    :param emissions: list of the names of the emissions, if any, defaults to None
    :type emissions: list, optional
    """
    if states is None:
        states = ['S_' + str(i) for i in range(model.n_states)]

    if emissions is None:
        emissions = create_emissions_name_list(model, hmm_type)

    print_startprob_table(model, states)
    print_transition_table(model, states)

    if hmm_type == 'Multinomial':
        print_emission_table(model, states, emissions)
    elif hmm_type == 'Gaussian':
        print_means_table(model, states, emissions)
        print_covars_table(model, states, emissions)
    elif hmm_type == 'Heterogeneous':
        print_means_table(model, states, emissions[0])
        print_covars_table(model, states, emissions[0])
        print_emission_table_het(model, states, emissions[1])
    return


def create_emissions_name_list(model, hmm_type='Multinomial'):
    """
    Helper method for the pretty print function. If the emissions
    are not given, it generates lists for the corresponding models.
    """
    if hmm_type == 'Multinomial':
        emissions = []
        for i in range(model.n_emissions):
            emissions.append(
                ['E_' + str(i) + str(j) for j in range(model.n_features[i])]
            )
    elif hmm_type == 'Gaussian':
        emissions = ['E_' + str(i) for i in range(model.n_emissions)]
    elif hmm_type == 'Heterogeneous':
        g_emissions = ['GE_' + str(i) for i in range(model.n_g_emissions)]
        d_emissions = []
        for i in range(model.n_d_emissions):
            d_emissions.append(
                ['DE_' + str(i) + str(j) for j in range(model.n_d_features[i])]
            )
        emissions = (g_emissions, d_emissions)
    return emissions


def print_table(rows, header):
    """
    Helper method for the pretty print function. It prints the parameters
    as a nice table.
    """
    t = PrettyTable(header)
    for row in rows:
        t.add_row(row)
    print(t)
    return


def print_startprob_table(model, states):
    """
    Helper method for the pretty print function. Prints the
    prior probabilities.
    """
    print('Priors')
    rows = []
    for i, sp in enumerate(model.pi):
        rows.append('P({})={:.3f}'.format(states[i], sp))
    print_table([rows], states)
    return


def print_transition_table(model, states):
    """
    Helper method for the pretty print function. Prints the state
    transition probabilities.
    """
    print('Transitions')
    rows = []
    for i, row in enumerate(model.A):
        rows.append(
            [states[i]]
            + [
                'P({}|{})={:.3f}'.format(states[j], states[i], tp)
                for j, tp in enumerate(row)
            ]
        )
    print_table(rows, ['_'] + states)
    return


def print_emission_table(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the
    emission probabilities.
    """
    print('Emissions')
    for e in range(model.n_emissions):
        rows = []
        for i, row in enumerate(model.B[e]):
            rows.append(
                [states[i]]
                + [
                    'P({}|{})={:.3f}'.format(emissions[e][j], states[i], ep)
                    for j, ep in enumerate(row)
                ]
            )
        print_table(rows, ['_'] + emissions[e])
    return


def print_emission_table_het(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the
    emission probabilities in case of a HeterogeneousHMM.
     """
    print('Emissions')
    for e in range(model.n_d_emissions):
        rows = []
        for i, row in enumerate(model.B[e]):
            rows.append(
                [states[i]]
                + [
                    'P({}|{})={:.3f}'.format(emissions[e][j], states[i], ep)
                    for j, ep in enumerate(row)
                ]
            )
        print_table(rows, ['_'] + emissions[e])
    return


def print_means_table(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the
    means of the GaussianHMM.
     """
    print('Means')
    rows = []
    for i, row in enumerate(model.means):
        rows.append([states[i]] + ['{:.3f}'.format(ep) for ep in row])
    print_table(rows, ['_'] + emissions)
    return


def print_covars_table(model, states, emissions):
    """
    Helper method for the pretty print function. Prints the
    covariances of the GaussianHMM.
     """
    print('Covariances')
    for ns, state in enumerate(states):
        print(state)
        rows = []
        for i, row in enumerate(model.covars[ns]):
            rows.append([emissions[i]] + ['{:.3f}'.format(ep) for ep in row])
        print_table(rows, ['_'] + emissions)
    return
