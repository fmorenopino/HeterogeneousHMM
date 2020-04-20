"""
Created on Nov 20, 2019

@author: semese

Parts of the code come from: https://github.com/hmmlearn/hmmlearn
"""
import numpy as np
import pickle
from scipy.special import logsumexp
from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.datasets import make_spd_matrix
from prettytable import PrettyTable
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib.lines import Line2D
import seaborn as sns

plt.style.use("seaborn-ticks")

COVARIANCE_TYPES = frozenset(("spherical", "tied", "diagonal", "full"))
COLORS = sns.color_palette("colorblind", n_colors=15)

# ---------------------------------------------------------------------------- #
#                            Utils for the HMM models                          #
# ---------------------------------------------------------------------------- #
def normalise(a, axis=None):
    """
    Normalises the input array so that it sums to 1.
    Args:
        a (array) - input data to be normalised
        axis (int) - dimension along which normalidation has to be performed
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        a_sum[a_sum == 0] = 1  # Make sure it's not divided by zero.
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape
    a /= a_sum
    return a


def log_normalise(a, axis=None):
    """
    normalises the input array so that ``sum(exp(a)) == 1``.
    Args:
        a (array) - Non-normalised input data.
        axis (int) - dimension along which normalization is performed.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


def log_mask_zero(a):
    """
    Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalised to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this un-harmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def check_if_attributes_set(model, attr=None):
    """
    Checks if the model attributes are set before training. This is only necessary
    if the 'no_init' option is selected, because in that case the model parameters
    are expected to be set apriori, and won't be reinitialised before training or
    if some of the discrete emission probabilities aren't trained.
    """
    params_dict = {"t": "A", "s": "pi", "e": "B", "m": "means", "c": "covars"}
    model_dict = model.__dict__
    if attr is not None:
        if not params_dict[attr] in model_dict.keys():
            raise AttributeError(
                "Attr self."
                + params_dict[attr]
                + " must be initialised before training"
            )
    else:
        for par in model.params:
            if params_dict[par] in model_dict.keys():
                continue
            else:
                raise AttributeError(
                    "Attr self."
                    + params_dict[par]
                    + " must be initialised before training"
                )
    return


# Copied from scikit-learn 0.19.
def validate_covars(covars, covariance_type, n_states):
    """Do basic checks on matrix covariance sizes and values."""

    if covariance_type == "spherical":
        if len(covars) != n_states:
            raise ValueError("'spherical' covars have length n_states")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == "tied":
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif not np.allclose(covars, covars.T) or np.any(linalg.eigvalsh(covars) <= 0):
            raise ValueError("'tied' covars must be symmetric, " "positive-definite")
    elif covariance_type == "diagonal":
        if len(covars.shape) != 2:
            raise ValueError("'diagonal' covars must have shape " "(n_states, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diagonal' covars must be non-negative")
    elif covariance_type == "full":
        if len(covars.shape) != 3:
            raise ValueError(
                "'full' covars must have shape " "(n_states, n_dim, n_dim)"
            )
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError(
                "'full' covars must have shape " "(n_states, n_dim, n_dim)"
            )
        for n, cv in enumerate(covars):
            if not np.allclose(cv, cv.T) or np.any(linalg.eigvalsh(cv) <= 0):
                raise ValueError(
                    "component %d of 'full' covars must be "
                    "symmetric, positive-definite" % n
                )
    else:
        raise ValueError(
            "covariance_type must be one of "
            + "'spherical', 'tied', 'diagonal', 'full'"
        )
    return


def init_covars(tied_cv, covariance_type, n_states):
    """
        Helper function for initialising the covariances based on the
        covariance type. See class definition for details.
    """
    if covariance_type == "spherical":
        cv = tied_cv.mean() * np.ones((n_states,))
    elif covariance_type == "tied":
        cv = tied_cv
    elif covariance_type == "diagonal":
        cv = np.tile(np.diag(tied_cv), (n_states, 1))
    elif covariance_type == "full":
        cv = np.tile(tied_cv, (n_states, 1, 1))
    return cv


def fill_covars(covars, covariance_type="full", n_states=1, n_features=1):
    """
    Return the covariance matrices in full form: (n_states, n_features, n_features)
    """
    new_covars = np.array(covars, copy=True)
    if covariance_type == "full":
        return new_covars
    elif covariance_type == "diagonal":
        return np.array(list(map(np.diag, new_covars)))
    elif covariance_type == "tied":
        return np.tile(new_covars, (n_states, 1, 1))
    elif covariance_type == "spherical":
        eye = np.eye(n_features)[np.newaxis, :, :]
        new_covars = new_covars[:, np.newaxis, np.newaxis]
        temp = eye * new_covars
        return temp


# This function is only used for testing
def make_covar_matrix(covariance_type, n_components, n_features, random_state=None):
    mincv = 0.1
    prng = check_random_state(random_state)
    if covariance_type == "spherical":
        return (mincv + mincv * prng.random_sample((n_components,))) ** 2
    elif covariance_type == "tied":
        return make_spd_matrix(n_features) + mincv * np.eye(n_features)
    elif covariance_type == "diagonal":
        return (mincv + mincv * prng.random_sample((n_components, n_features))) ** 2
    elif covariance_type == "full":
        return np.array(
            [
                (
                    make_spd_matrix(n_features, random_state=prng)
                    + mincv * np.eye(n_features)
                )
                for x in range(n_components)
            ]
        )


# ---------------------------------------------------------------------------- #
#                             Visualisation utils                              #
# ---------------------------------------------------------------------------- #
def plot_log_likelihood_evolution(log_likelihood, filename=None):
    """
        Plot the evolution of the log-likelihood over the iterations.
        Args:
            log_likelihood (list) - the list of log-likelihood values
            filename (str) - full path to where to save the figure
    """
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes()
    iters = np.arange(1, len(log_likelihood) + 1)
    ax.plot(iters, log_likelihood)
    ax.set_xlabel("# iterations")
    ax.set_ylabel("Log-likelihood")

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()

    return


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
        Args:
            obs_seq (ndarray) - an observation sequence
            data_cols (list) - name of (Gaussian) data columns (name of each
                column)
            state_seq (list) - the decoded state sequence
            discrete_columns (list, optional) - only in case of HMM with labels, the
                list of label names
            state_names (list, optional) - list of state names, only if applicable
            time_stamps (list, optional) - list of time_stamps when the observations where
                recorded (if applicable, otherwise discrete values are applied, starting
                of 0 to len(state_seq))
            figsize (tuple, optional) - size of the plot
            filename (str) - full path with figure name if it should be saved
    """
    if len(data_cols) > 1:
        fig, axes = plt.subplots(nrows=len(data_cols), figsize=figsize)
        for i, (ax, variable) in enumerate(zip(axes, data_cols)):
            if discrete_columns is not None and variable in discrete_columns:
                ax.set_ylim(
                    [np.amin(obs_seq[:, i]) - 0.1, np.amax(obs_seq[:, i]) + 0.1]
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
        color="k",
        marker=".",
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
        ax.set_xticklabels([ts.strftime("%m/%d/%Y, %H:%M") for ts in time_stamps])
    # Add the legend.
    # ax.legend(loc="upper right", handletextpad=0)
    ax.set_title(variable)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


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
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=True,
            )
        else:
            ax.legend(
                custom_lines,
                ["State_" + str(unq) for unq in unique],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=True,
            )


# ---------------------------------------------------------------------------- #
#                                   IO utils                                   #
# ---------------------------------------------------------------------------- #
def pickle_model(model, filename):
    """
        Function to save an hmm model (or any variable) to a pickle file.
        Args:
            model (object) - the variable to be saved
            filename (str) - full path or just file name where to save the variable
    """
    with open(filename, "wb2") as f:
        pickle.dump(model, f)
    return


def read_pickled_model(filename):
    """
        Function to load an hmm model (or any variable) from a pickle file.
        Args:
            filename (str) - full path or just file name where to save the variable
        Returns:
            model (object) - the variable from the pickle file
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def pretty_print_hmm(model, hmm_type="Multinomial", states=None, emissions=None):
    """
        Function to pretty print the parameters of an hmm model.
        Args:
            model (object) - and HMM object
            hmm_type (str) - the type of the HMM model; can be "Multinomial",
                "Gaussian" or "Heterogeneous".
            states (list, optional) - list with the name of states, if any
            emissions (list, optional) - list of the names of the emissions, if any
    """
    if states is None:
        states = ["S_" + str(i) for i in range(model.n_states)]

    if emissions is None:
        emissions = create_emissions_name_list(model, hmm_type)

    print_startprob_table(model, states)
    print_transition_table(model, states)

    if hmm_type == "Multinomial":
        print_emission_table(model, states, emissions)
    elif hmm_type == "Gaussian":
        print_means_table(model, states, emissions)
        print_covars_table(model, states, emissions)
    elif hmm_type == "Heterogeneous":
        print_means_table(model, states, emissions[0])
        print_covars_table(model, states, emissions[0])
        print_emission_table_het(model, states, emissions[1])
    return


def create_emissions_name_list(model, hmm_type="Multinomial"):
    """
        Helper method for the pretty print function. If the emissions
        are not given, it generates lists for the corresponding models.
        Args:
            model (object) - an HMM model
            hmm_type (str) - the type of the HMM; see pretty_print_hmm
        Returns:
            emissions (list or tuple) - the generated emission labels
    """
    if hmm_type == "Multinomial":
        emissions = []
        for i in range(model.n_emissions):
            emissions.append(
                ["E_" + str(i) + str(j) for j in range(model.n_features[i])]
            )
    elif hmm_type == "Gaussian":
        emissions = ["E_" + str(i) for i in range(model.n_emissions)]
    elif hmm_type == "Heterogeneous":
        g_emissions = ["GE_" + str(i) for i in range(model.n_g_emissions)]
        d_emissions = []
        for i in range(model.n_d_emissions):
            d_emissions.append(
                ["DE_" + str(i) + str(j) for j in range(model.n_d_features[i])]
            )
        emissions = (g_emissions, d_emissions)
    return emissions


def print_table(rows, header):
    """
        Helper method for the pretty print function. It prints the parameters
        as a nice table.
        Args:
            rows (list) - the rows of the table
            header (list) - the header of the table
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
        Args:
            model (object) - an HMM model
            states (list) - the list of state names
    """
    print("Priors")
    rows = []
    for i, sp in enumerate(model.pi):
        rows.append("P({})={:.3f}".format(states[i], sp))
    print_table([rows], states)
    return


def print_transition_table(model, states):
    """
        Helper method for the pretty print function. Prints the state
        transition probabilities.
        Args:
            model (object) - an HMM model
            states (list) - the list of state names
    """
    print("Transitions")
    rows = []
    for i, row in enumerate(model.A):
        rows.append(
            [states[i]]
            + [
                "P({}|{})={:.3f}".format(states[j], states[i], tp)
                for j, tp in enumerate(row)
            ]
        )
    print_table(rows, ["_"] + states)
    return


def print_emission_table(model, states, emissions):
    """
        Helper method for the pretty print function. Prints the
        emission probabilities.
        Args:
            model (object) - an HMM model
            states (list) - the list of state names
            emissions (list) - the list of emission names
    """
    print("Emissions")
    for e in range(model.n_emissions):
        rows = []
        for i, row in enumerate(model.B[e]):
            rows.append(
                [states[i]]
                + [
                    "P({}|{})={:.3f}".format(emissions[e][j], states[i], ep)
                    for j, ep in enumerate(row)
                ]
            )
        print_table(rows, ["_"] + emissions[e])
    return


def print_emission_table_het(model, states, emissions):
    """
         Helper method for the pretty print function. Prints the
         emission probabilities in case of a HeterogeneousHMM.
         Args:
             model (object) - an HMM model
             states (list) - the list of state names
             emissions (list) - the list of emission names
     """
    print("Emissions")
    for e in range(model.n_d_emissions):
        rows = []
        for i, row in enumerate(model.B[e]):
            rows.append(
                [states[i]]
                + [
                    "P({}|{})={:.3f}".format(emissions[e][j], states[i], ep)
                    for j, ep in enumerate(row)
                ]
            )
        print_table(rows, ["_"] + emissions[e])
    return


def print_means_table(model, states, emissions):
    """
         Helper method for the pretty print function. Prints the
         means of the GaussianHMM.
         Args:
             model (object) - an HMM model
             states (list) - the list of state names
             emissions (list) - the list of emission names
     """
    print("Means")
    rows = []
    for i, row in enumerate(model.means):
        rows.append([states[i]] + ["{:.3f}".format(ep) for ep in row])
    print_table(rows, ["_"] + emissions)
    return


def print_covars_table(model, states, emissions):
    """
         Helper method for the pretty print function. Prints the
         covariances of the GaussianHMM.
         Args:
             model (object) - an HMM model
             states (list) - the list of state names
             emissions (list) - the list of emission names
     """
    print("Covariances")
    for ns, state in enumerate(states):
        print(state)
        rows = []
        for i, row in enumerate(model.covars[ns]):
            rows.append([emissions[i]] + ["{:.3f}".format(ep) for ep in row])
        print_table(rows, ["_"] + emissions)
    return
