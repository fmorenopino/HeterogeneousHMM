import numpy as np
import csv
import copy
from .utils import print_table

# ---------------------------------------------------------------------------- #
#            Generating training samples and computind data info               #
# ---------------------------------------------------------------------------- #
def generate_samples(model, n_seq, n_samples):
    """
    Generates samples from a hmmlearn model and arranges them to the
    correct input forms for training.
    Args:
        model (hmmlearn) - an hmmlearn model
        n_seq (int) - number of observation sequences to generate
        n_samples (int) - number of total samples to generate
    Returns:
        observations (list) - list of observation sequences which are arrays
        observations_hmmlearn (array) - array of concatenated observation sequences
        lengths (list) - list of individual obseervation sequence lengths (it
            sums up to n_samples)
    """

    def add_new_samples(X, nsq, observations, observations_hmmlearn, lengths):
        observations.append(np.array([X[i].tolist() for i in range(len(X))]))
        if nsq == 0:
            observations_hmmlearn = X
        else:
            observations_hmmlearn = np.concatenate((observations_hmmlearn, X), axis=0)
        lengths.append(len(X))
        return observations, observations_hmmlearn, lengths

    observations = []
    observations_hmmlearn = []
    lengths = []
    n_samples_per_seq = int(n_samples / n_seq)
    if n_samples % n_seq == 0:
        for nsq in range(n_seq):
            X, _ = model.sample(n_samples_per_seq)
            observations, observations_hmmlearn, lengths = add_new_samples(
                X, nsq, observations, observations_hmmlearn, lengths
            )
    else:
        n_samples_acc = 0
        for nsq in range(n_seq - 1):
            X, _ = model.sample(n_samples_per_seq)
            observations, observations_hmmlearn, lengths = add_new_samples(
                X, nsq, observations, observations_hmmlearn, lengths
            )
            n_samples_acc += n_samples_per_seq
        n_samples_per_seq = n_samples - n_samples_acc
        X, _ = model.sample(n_samples_per_seq)
        observations, observations_hmmlearn, lengths = add_new_samples(
            X, nsq, observations, observations_hmmlearn, lengths
        )

    return observations, observations_hmmlearn, lengths


def introduce_nans(observations, prob=0.2, nan_type="float", missing_value=-1):
    """
    Function to introduce missing observations in the observation sequences.
    Args:
        observations (list) - list of arrays which represent observation sequences
        prob (float, optional) - the percentage of missing data to introduce
        nan_type (str, optional) - since python doesn't support numpy nan for
            integer cases, for that we introduce -1
    """
    missing_observations = copy.deepcopy(observations)
    for obs_idx, observation in enumerate(observations):
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if np.random.rand() < prob:
                    if nan_type == "integer":
                        missing_observations[obs_idx][i][j] = missing_value
                    else:
                        missing_observations[obs_idx][i][j] = np.nan
    return missing_observations


def obs_sequences_to_array(obs_sequences):
    """
        Stack the observation sequences into one array.
    """
    data = np.vstack([obs for obs in obs_sequences])
    return data


def compute_metadata(data):
    """
        Compute the metadata of the test cases and return as a list.
    """
    # counting partially missing observations
    partially_missing = sum(
        [True for row in data if any(np.isnan(row)) and not all(np.isnan(row))]
    )
    p_partially_missing = (partially_missing * 100.0) / data.shape[0]
    # counting completely missing observations
    completely_missing = sum([True for row in data if all(np.isnan(row))])
    p_completely_missing = (completely_missing * 100.0) / data.shape[0]
    # full observations
    full = data.shape[0] - (partially_missing + completely_missing)
    return [
        full,
        partially_missing,
        p_partially_missing,
        completely_missing,
        p_completely_missing,
    ]


# ---------------------------------------------------------------------------- #
#                                   IO utils                                   #
# ---------------------------------------------------------------------------- #
def write_data_to_csv(filename, header, data):
    """
    Function to write data to a csv file.
    Args:
        filename (str) - the full path or just file name to use
        header (list) - a list of strings to use as the header of the csv
        data (2D list) - with each element being a list of data that
            corresponds to a row in the csv.
    """
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    return


def print_hmmlearn_parameters(
    model, hmm_type="Multinomial", states=None, emissions=None
):
    """
        Function to pretty print the parameters of an hmmlearn model.
        Args:
            model (object) - and hmmlearn object
            hmm_type (str) - the type of the HMM model; can be "Multinomial" or
                "Gaussian"
            states (list, optional) - list with the name of states, if any
            emissions (list, optional) - list of the names of the emissions, if any
    """
    if states is None:
        states = ["State_" + str(i) for i in range(model.n_components)]

    print("Priors")
    rows = []
    for i, sp in enumerate(model.startprob_):
        rows.append("P({})={:.3f}".format(i, sp))
    print_table([rows], states)

    print("Transitions")
    rows = []
    for i, row in enumerate(model.transmat_):
        rows.append(
            [states[i]]
            + ["P({}|{})={:.3f}".format(j, i, tp) for j, tp in enumerate(row)]
        )
    print_table(rows, ["_"] + states)

    if hmm_type == "Multinomial":
        if emissions is None:
            emissions = [str(i) for i in range(model.n_features)]
        print("Emissions")
        rows = []
        for i, row in enumerate(model.emissionprob_):
            rows.append(
                [states[i]]
                + ["P({}|{})={:.3f}".format(j, i, ep) for j, ep in enumerate(row)]
            )
        print_table(rows, ["_"] + emissions)

    elif hmm_type == "Gaussian":
        if emissions is None:
            emissions = ["Emission_" + str(i) for i in range(model.n_features)]

        print("Means")
        rows = []
        for i, row in enumerate(model.means_):
            rows.append([states[i]] + ["{:.3f}".format(ep) for ep in row])
        print_table(rows, ["_"] + emissions)

        print("Covariances")
        for ns, state in enumerate(states):
            print(state)
            rows = []
            for i, row in enumerate(model.covars_[ns]):
                rows.append([emissions[i]] + ["{:.3f}".format(ep) for ep in row])
            print_table(rows, ["_"] + emissions)
