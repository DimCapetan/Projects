import pybop
import pybamm
import os
import numpy as np
import pandas as pd


# We will use the number of parameters that will be used, the parameter set, the name of the cost function and the optimiser, as well as the current that will be added
def pybop_wrapper(data_file: str, data_file2: str, param_set: str, model_selection: str, parameters: list,
                  parameters2: list, cost_function: str, optimiser: str, temp: bool, solver_name: str = "IDAKLU",
                  solver_kwargs: dict | None = None, model_options_override: dict | None = None,
                  merge_strategy: str = "override", extra_model_kwargs: dict | None = None,
                  solver_name2: str = "IDAKLU",
                  solver_kwargs2: dict | None = None, model_options_override2: dict | None = None,
                  merge_strategy2: str = "override", extra_model_kwargs2: dict | None = None, **kwargs):
    '''
    Runs a full PyBOP parameter optimisation
    
    Args: 
        data_file: directory of the data that will be used in the optimisation process.
        param_set: name of the PyBOP parameter set (e.g. Chen2020).
        model_selection: name of the physics based battery model (e.g. DFN, DFN_Thermal, SPM, etc.).
        parameters: parameter names to optimise (e.g. ["c_max", "h"]).
        cost_function: name of a PyBOP cost function (e.g. RMSE, MSE, MAE, etc.).
        optimiser: name of a PyBOP optimiser (e.g. "PSO", "NelderMead", etc.).
        sigma: noise-level or initial step-size for your cost/optimiser.
        
        Returns:
        result: the "OptimisationResult" from PyBOP
    '''

    # Create an absolute path for the data
    dataset_path = data_file

    # importing the experimental/synthetic data
    df = pd.read_excel(dataset_path)
    df = df.iloc[::10].reset_index(drop=True)

    # With the data that we have from the excel file, build the dataset
    dataset = pybop.Dataset(
        {
            "Time [s]": df["Time [s]"].to_numpy(),
            "Current function [A]": df["Current [A]"].to_numpy(),
            "Voltage [V]": df["Voltage [V]"].to_numpy()
        })

    # Assign a signal so PyBOP can see exactly what curve the model is trying to fit when optimising parameters
    signal = ["Voltage [V]"]

    # Create a dict for the different parameter sets that the user might enter
    param_set_dict = {
        "Ai2020": pybop.ParameterSet("Ai2020"),
        "Chayambuka2022": pybop.ParameterSet("Chayambuka2022"),
        "Chen2020": pybop.ParameterSet("Chen2020"),
        "Chen2020_composite": pybop.ParameterSet("Chen2020_composite"),
        "ECM_Example": pybop.ParameterSet("ECM_Example"),
        "Ecker2015": pybop.ParameterSet("Ecker2015"),
        "Ecker2015_graphite_halfcell": pybop.ParameterSet("Ecker2015_graphite_halfcell"),
        "MSMR_Example": pybop.ParameterSet("MSMR_Example"),
        "Marquis2019": pybop.ParameterSet("Marquis2019"),
        "Mohtat2020": pybop.ParameterSet("Mohtat2020"),
        "NCA_Kim2011": pybop.ParameterSet("NCA_Kim2011"),
        "OKane2022": pybop.ParameterSet("OKane2022"),
        "OKane2022_graphite_SiOx_halfcell": pybop.ParameterSet("OKane2022_graphite_SiOx_halfcell"),
        "ORegan2022": pybop.ParameterSet("ORegan2022"),
        "Prada2013": pybop.ParameterSet("Prada2013"),
        "Ramadass2004": pybop.ParameterSet("Ramadass2004"),
        "Sulzer2019": pybop.ParameterSet("Sulzer2019"),
        "Xu2019": pybop.ParameterSet("Xu2019")
    }

    if param_set not in param_set_dict:
        raise ValueError(f"The parameter set {param_set} does not exist.",
                         f"These are the available options for parameter sets {list(param_set_dict)}")
    else:
        parameterSet = param_set_dict[param_set]

    if param_set == "Marquis2019":
        # Function for the diffusivities of the positive, negative electrodes and the electrolyte
        def graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T):

            # D_ref = 3.9e-14
            D_ref = pybamm.Parameter("Negative particle diffusivity pre-exponential [m2.s-1]")
            E_D_s = 42770
            arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return D_ref * arrhenius

        def lico2_diffusivity_Dualfoil1998(sto, T):

            # D_ref = 1e-13
            D_ref = pybamm.Parameter("Positive particle diffusivity pre-exponential [m2.s-1]")
            E_D_s = 18550
            arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return D_ref * arrhenius

        D_c_e_base = 5.34e-10 * np.exp(-0.65 * 1.0)
        sigma_e_base = 0.0911 + 1.9101 - 1.052 + 0.1554

        def electrolyte_diffusivity_Capiglia1999(c_e, T):

            D_c_e = 5.34e-10 * np.exp(-0.65 * c_e / 1000)
            D_ce = pybamm.Parameter("Electrolyte diffusivity pre-exponential [m2.s-1]")
            E_D_e = 37040
            arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return (D_ce / D_c_e_base) * D_c_e * arrhenius

        # Now calling the function for the conductivity of the electrolyte
        def electrolyte_conductivity_Capiglia1999(c_e, T):

            sigma_e = (0.0911 + 1.9101 * (c_e / 1000) - 1.052 * (c_e / 1000) ** 2 + 0.1554 * (c_e / 1000) ** 3)
            sigmae = pybamm.Parameter("Electrolyte conductivity pre-exponential [S.m-1]")

            E_k_e = 34700
            arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return (sigmae / sigma_e_base) * sigma_e * arrhenius

        def graphite_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, c_s_max, T):
            # m_ref = 2 * 10 ** (-5)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
            m_ref = pybamm.Parameter("Negative electrode reaction rate (A/m2)(m3/mol)**1.5")
            E_r = 37480
            arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5

        def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, c_s_max, T):

            # m_ref = 6 * 10 ** (-7)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
            m_ref = pybamm.Parameter("Positive electrode reaction rate (A/m2)(m3/mol)**1.5")
            E_r = 39570
            arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

            return m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5

        if param_set in {"Chen2020", "Marquis2019"}:
            def graphite_LGM50_ocp_Chen2020(sto):
                u_eq = (1.9793 * pybamm.exp(-39.3631 * sto) + 0.2482 - 0.0909 * pybamm.tanh(
                    29.8538 * (sto - 0.1234)) - 0.04478 * pybamm.tanh(14.9159 * (sto - 0.2769)) - 0.0205 * pybamm.tanh(
                    30.4444 * (sto - 0.6103)))
                return u_eq

            def nmc_LGM50_ocp_Chen2020(sto):
                u_eq = (-0.8090 * sto + 4.4875 - 0.0428 * pybamm.tanh(18.5138 * (sto - 0.5542)) - 17.7326 * pybamm.tanh(
                    15.7890 * (sto - 0.3117)) + 17.5842 * pybamm.tanh(15.9308 * (sto - 0.3120)))
                return u_eq

    # Create a buffer for the voltage to ensure that there is no sudden cut-off
    buffer = 0.5

    # Now create the parameter set and set cut-off voltages boundaries
    parameter_set = param_set_dict[param_set]
    parameter_set["Lower voltage cut-off [V]"] = min(df["Voltage [V]"]) - buffer
    parameter_set["Upper voltage cut-off [V]"] = max(df["Voltage [V]"]) + buffer

    if param_set in ["Chen2020", "Marquis2019"]:
        parameterSet.update({
            "Lower voltage cut-off [V]": min(df["Voltage [V]"]) - buffer,
            "Upper voltage cut-off [V]": max(df["Voltage [V]"]) + buffer,
            "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
            "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
            "Negative particle diffusivity [m2.s-1]": graphite_mcmb2528_diffusivity_Dualfoil1998,
            "Positive particle diffusivity [m2.s-1]": lico2_diffusivity_Dualfoil1998,
            "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Capiglia1999,
            "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Capiglia1999,
            "Negative electrode exchange-current density [A.m-2]": graphite_electrolyte_exchange_current_density_Dualfoil1998,
            "Positive electrode exchange-current density [A.m-2]": lico2_electrolyte_exchange_current_density_Dualfoil1998
        })

        parameterSet.update({
            "Negative particle diffusivity pre-exponential [m2.s-1]": 3.9e-14,
            "Positive particle diffusivity pre-exponential [m2.s-1]": 1e-13,
            "Electrolyte diffusivity pre-exponential [m2.s-1]": 5.34e-10 * np.exp(-0.65 * 1),
            "Electrolyte conductivity pre-exponential [S.m-1]": (
                        0.0911 + 1.9101 * (1) - 1.052 * (1) ** 2 + 0.1554 * (1) ** 3),
            "Negative electrode reaction rate (A/m2)(m3/mol)**1.5": 2e-5,
            "Positive electrode reaction rate (A/m2)(m3/mol)**1.5": 6e-7
        },
            check_already_exists=False)

    # Choose the solver for the problem

    if solver_kwargs is None:
        solver_kwargs = {}

    solver_kwargs.setdefault("rtol", 1e-6)
    solver_kwargs.setdefault("atol", 1e-6)
    solver_kwargs.setdefault("root_method", "casadi")

    kw = dict(solver_kwargs)

    solver_key = solver_name.strip().lower()
    if solver_key == "idaklu":
        kw.setdefault("root_method", "casadi")
        solver = pybamm.IDAKLUSolver(**kw)
    elif solver_key == "casadi":
        kw.setdefault("mode", "fast")
        solver = pybamm.CasadiSolver(**kw)
    elif solver_key == "jax":
        solver = pybamm.JaxSolver(**kw)
    else:
        raise ValueError(f"Unsupported solver_name {solver_name}."
                         f"Supported: IDAKLU, Casadi, JAX")

    # Create options for distribution of the particles
    distr_options = {
        "particle": "Fickian diffusion",
        "particle shape": "spherical",
        "surface form": "differential"
    }

    # Define the thermal model
    thermal_options_lumped = {
        "cell geometry": "pouch",
        "thermal": "lumped",
        "contact resistance": "true"
    }

    # Now that we have defined the parameter set and the parameters that we want to instantiate and seach for the appropriate model
    model_dict = {
        "DFN": (pybop.lithium_ion.DFN, {**distr_options}),
        "DFN_Thermal": (pybop.lithium_ion.DFN, {**distr_options, **thermal_options_lumped}),
        "SPM": (pybop.lithium_ion.SPM, {**distr_options}),
        "SPM_Thermal": (pybop.lithium_ion.SPM, {**distr_options, **thermal_options_lumped}),
        "SPMe": (pybop.lithium_ion.SPMe, {**distr_options}),
        "SPMe_Thermal": (pybop.lithium_ion.SPMe, {**distr_options, **thermal_options_lumped, })
    }

    if model_selection not in model_dict:
        raise ValueError(f"The model {model_selection} does not exist.",
                         f"These are the available options: {list(model_dict.keys())}")

    # Now set initial state for the model
    # initial_SoC = 1
    initial_state = {"Initial open-circuit voltage [V]": df["Voltage [V]"].iloc[0]}
    model_class, model_options = model_dict[model_selection]

    # Create an option to merge user-supplied model options if they are provided
    if model_options_override:
        if merge_strategy == "override":
            # user keys will replace the base keys
            model_options = {**model_options, **model_options_override}
        elif merge_strategy == "add_only":
            # add only the new keys
            for k, v in model_options_override.items():
                if k not in model_options:
                    model_options[k] = v
        elif merge_strategy == "replace":
            model_options = dict(model_options_override)
        else:
            raise ValueError(f"Unknown merge_strategy '{merge_strategy}'. Use 'override', 'add_only or 'replace'.")

    # Extra constructor kwargs (forwarded)
    if extra_model_kwargs is None:
        extra_model_kwargs = {}

    model = model_class(parameter_set=parameterSet, solver=solver, options=model_options, **extra_model_kwargs)

    # model.set_initial_state(initial_state)

    # Now that we defined and chose the parameter set, we will create a map for the different parameters that are optimised in this stage
    parameter_dict = {
        "cn_max": parameterSet["Maximum concentration in negative electrode [mol.m-3]"],
        "cp_max": parameterSet["Maximum concentration in positive electrode [mol.m-3]"],
        "cn_init": parameterSet["Initial concentration in negative electrode [mol.m-3]"],
        "cp_init": parameterSet["Initial concentration in positive electrode [mol.m-3]"],
        "h": parameterSet["Electrode height [m]"],
        "w": parameterSet["Electrode width [m]"],
        "n_thick": parameterSet["Negative electrode thickness [m]"],
        "p_thick": parameterSet["Positive electrode thickness [m]"],
        "n_act": parameterSet["Negative electrode active material volume fraction"],
        "p_act": parameterSet["Positive electrode active material volume fraction"],
        "n_por": parameterSet["Negative electrode porosity"],
        "p_por": parameterSet["Positive electrode porosity"],
        "n_rad": parameterSet["Negative particle radius [m]"],
        "p_rad": parameterSet["Positive particle radius [m]"]
    }

    # Now create a dictionary with just the names of the parameters
    param_name_dict = {
        "cn_max": "Maximum concentration in negative electrode [mol.m-3]",
        "cp_max": "Maximum concentration in positive electrode [mol.m-3]",
        "cn_init": "Initial concentration in negative electrode [mol.m-3]",
        "cp_init": "Initial concentration in positive electrode [mol.m-3]",
        "h": "Electrode height [m]",
        "w": "Electrode width [m]",
        "n_thick": "Negative electrode thickness [m]",
        "p_thick": "Positive electrode thickness [m]",
        "n_act": "Negative electrode active material volume fraction",
        "p_act": "Positive electrode active material volume fraction",
        "n_por": "Negative electrode porosity",
        "p_por": "Positive electrode porosity",
        "n_rad": "Negative particle radius [m]",
        "p_rad": "Positive particle radius [m]"
    }

    if temp == True:
        temp = parameterSet["Initial temperature [K]"]
    else:
        parameterSet["Initial temperature [K]"] = int(input("Please enter the new Initial Temperature in K: "))

    # Make sure that the parameters is a list of strings
    if isinstance(parameters, str):
        param_list = [parameters]
    else:
        param_list = list(parameters)

    # Raise an error if another output has been placed into the function
    for key in param_list:
        if key not in parameter_dict:
            raise ValueError(f"The parameter set {parameters} does not exist.",
                             f"These are the available options for parameter sets {list(parameter_dict.keys())}")

    param_keys = {"cn_max", "cp_max", "cn_init", "cp_init", "h", "w", "n_thick", "p_thick", "n_act", "p_act", "n_por",
                  "p_por", "n_rad", "p_rad"}

    optim_keys = [k for k in param_list if k in param_keys]

    if not optim_keys:
        raise ValueError("No valid parameters found for either stage")

    # Now we will create the map for the cost functions and optimisers
    cost_dict = {
        "Minkowski": lambda problem, **kwargs: pybop.Minkowski(problem=problem, **kwargs),
        "Log Posterior": lambda problem, **kwargs: pybop.LogPosterior(
            log_likelihood=pybop.GaussianLogLikelihoodKnownSigma(problem=problem, **kwargs)),
        "MAE": lambda problem: pybop.MeanAbsoluteError(problem=problem),
        "MSE": lambda problem: pybop.MeanSquaredError(problem=problem),
        "RMSE": lambda problem: pybop.RootMeanSquaredError(problem=problem),
        "SoP": lambda problem, **kwargs: pybop.SumOfPower(problem=problem, **kwargs),
        "SSE": lambda problem: pybop.SumSquaredError(problem=problem),
        "Gaussian Log Likelihood": lambda problem, **kwargs: pybop.GaussianLogLikelihoodKnownSigma(problem=problem,
                                                                                                   **kwargs)
    }

    # Now raise an error if another output has been placed into the function
    if cost_function not in cost_dict:
        raise ValueError(f"The parameter set {cost_function} does not exist.",
                         f"These are the available options for parameter sets {list(cost_dict)}")

    # Now we will do the same with the optimiser
    optim_dict = {
        "CMAES": lambda cost, **kwargs: pybop.CMAES(cost, **kwargs),
        "CuckooSearch": lambda cost, **kwargs: pybop.CuckooSearch(cost, **kwargs),
        "DifferentialEvolution": lambda cost, **kwargs: pybop.SciPyDifferentialEvolution(cost, **kwargs),
        "SciPyMinimize": lambda cost, **kwargs: pybop.SciPyMinimize(cost, **kwargs),
        "XNES": lambda cost, **kwargs: pybop.XNES(cost, **kwargs),
        "GradientDescent": lambda cost, **kwargs: pybop.GradientDescent(cost, **kwargs),
        "IRPropMin": lambda cost, **kwargs: pybop.IRPropMin(cost, **kwargs),
        "NelderMead": lambda cost, **kwargs: pybop.NelderMead(cost, **kwargs),
        "PSO": lambda cost, **kwargs: pybop.PSO(cost, **kwargs),
        "RandomSearch": lambda cost, **kwargs: pybop.RandomSearch(cost, **kwargs),
        "SNES": lambda cost, **kwargs: pybop.SNES(cost, **kwargs),
        "SimulatedAnnealing": lambda cost, **kwargs: pybop.SimulatedAnnealing(cost, **kwargs),
        "AdamW": lambda cost, **kwargs: pybop.AdamW(cost, **kwargs)
    }

    # Now raise an error if another output has been placed into the function
    if optimiser not in optim_dict:
        raise ValueError(f"The optimiser {optimiser} does not exist.",
                         f"These are the available options: {list(optim_dict)}")

    fitting_parameters = []
    bounds_l = []
    bounds_u = []
    for key in optim_keys:
        # assign the key to a variable so it can capture the name of the parameter
        val = parameter_dict[key]
        # Now assign the lower and the upper boundary for the parameter
        if key in {"cn_max", "cp_max", "cn_init", "cp_init"}:
            lower, upper = 0.8 * val, 1.2 * val
        elif key == "h":
            tol = 0.005
            lower, upper = val - tol, val + tol
        elif key == "w":
            tol = 0.03
            lower, upper = val - tol, val + tol
        elif key in {"n_thick", "p_thick"}:
            tol = 3e-6
            lower, upper = val - tol, val + tol
        elif key in {"n_rad", "p_rad"}:
            tol = 2e-5
            lower, upper = val - tol, val + tol
        elif key == "n_act":
            lower, upper = 0.5 * val, 1 - (1.1 * parameter_dict["n_por"])
        elif key == "p_act":
            lower, upper = 0.5 * val, 1 - (1.1 * parameter_dict["p_por"])
        elif key in {"n_por", "p_por"}:
            lower, upper = 0.9 * val, 1.1 * val
        # Finally append the parameter that is created to the new parameter list, to gather the list with all the parameters that will undergo optimisation
        if key in {"n_act", "p_act"}:
            fitting_parameters.append(
                pybop.Parameter(
                    param_name_dict[key],
                    bounds=[lower, upper],
                    initial_value=val * 0.8
                )
            )
        else:
            fitting_parameters.append(
                pybop.Parameter(
                    param_name_dict[key],
                    bounds=[lower, upper],
                    initial_value=val
                )
            )
        bounds_l.append(lower)
        bounds_u.append(upper)

    lower_bounds = np.array(bounds_l)
    upper_bounds = np.array(bounds_u)
    sigma0 = 0.1 * (upper_bounds - lower_bounds)

    params = pybop.Parameters(*fitting_parameters)
    problem = pybop.FittingProblem(model=model, parameters=params, dataset=dataset, signal=signal,
                                   initial_state=model.set_initial_state(initial_state))
    cost = cost_dict[cost_function](problem)
    optim = optim_dict[optimiser](
        cost,
        verbose=True,
        sigma0=sigma0,
        verbose_print_rate=1,
        allow_infeasible_solutions=False,
        max_iterations=2000,
        max_unchanged_iterations=60,
        parallel=True
    )

    results = optim.run()

    cost_value = results.final_cost
    time = results.time
    iterations = results.n_iterations

    # Create an initial values list and then evaluate the problem with these values
    init_values = {k: parameter_dict[k] for k in optim_keys}
    optimised_values = list(results.x)

    print(" ")
    print(
        "--------------------------------------------------------------- STAGE 1 OPTIMISATION RESULTS ---------------------------------------------------------------")

    for short_key, opt_val in zip(optim_keys, optimised_values):
        init_val = init_values[short_key]
        long_name = param_name_dict[short_key]

        print(f"{short_key}: {init_val} -> {opt_val}  ({long_name})")

    print(f"Stage1 final cost: {cost_value}")
    print(f"Stage1 iterations: {iterations}")
    print(f"Stage1 time (s): {time}")
    print(" ")

    # Plot the graph to compare the results of the initial and the optimised parameters
    init_vector = [init_values[k] for k in optim_keys]
    pred_init = problem.evaluate(init_vector)
    pred_opt = problem.evaluate(results.x)

    t = dataset["Time [s]"]
    V = dataset["Voltage [V]"]

    pybop.plot.trajectories(
        t,
        [
            V,
            pred_init["Voltage [V]"],
            pred_opt["Voltage [V]"]
        ],
        trace_names=["Reference Model", "Initial Model", "Optimised Model"],
        xaxis_title="Time / s",
        yaxis_title="Voltage / V"
    )

    # Now updating the optimised parameters before going to the second stage of the optimisation process

    updated_params = {}
    for key, opt_val in zip(optim_keys, results.x):
        long_name = param_name_dict[key]
        parameterSet[long_name] = opt_val
        updated_params[long_name] = opt_val

    if len(updated_params) != len(optim_keys):
        raise RuntimeError("Optimiser output length mismatch.")

    parameterSet.update(updated_params, check_already_exists=False)

    # ------------------------------------------------------------------ STAGE 2 ------------------------------------------------------------------ 

    if isinstance(parameters2, str):
        stage2_requested = [parameters2]
    elif parameters2 is None:
        stage2_requested = []  # That means skip stage 2 completely
    else:
        stage2_requested = list(parameters2)

    # Define Stage 2 candidate parameters: short key -> (long name, default value if missing)
    stage2_candidates = {
        "n_diff": ("Negative particle diffusivity pre-exponential [m2.s-1]",
                   parameterSet["Negative particle diffusivity pre-exponential [m2.s-1]"]),
        "p_diff": ("Positive particle diffusivity pre-exponential [m2.s-1]",
                   parameterSet["Positive particle diffusivity pre-exponential [m2.s-1]"]),
        "e_diff": ("Electrolyte diffusivity pre-exponential [m2.s-1]",
                   parameterSet["Electrolyte diffusivity pre-exponential [m2.s-1]"]),
        "e_cond": ("Electrolyte conductivity pre-exponential [S.m-1]",
                   parameterSet["Electrolyte conductivity pre-exponential [S.m-1]"]),
        "i0_n": ("Negative electrode reaction rate (A/m2)(m3/mol)**1.5",
                 parameterSet["Negative electrode reaction rate (A/m2)(m3/mol)**1.5"]),
        "i0_p": ("Positive electrode reaction rate (A/m2)(m3/mol)**1.5",
                 parameterSet["Positive electrode reaction rate (A/m2)(m3/mol)**1.5"])
    }

    # Now filter the requested keys, if supplied. If the list is empty, it means that the second stage can be skipped
    if stage2_requested:
        stage2_keys = [k for k in stage2_requested if k in stage2_candidates]
        unused = set(stage2_requested) - set(stage2_keys)
        if unused:
            print(f"[Stage2] Ignoring unknown transport parameters: {unused}")
    else:
        stage2_keys = []  # Stage 2 can be skipped
    results2 = None
    updated_params_stage2 = {}

    if stage2_keys:

        for sk in stage2_keys:
            long_name, default_val = stage2_candidates[sk]
            parameterSet.update({long_name: default_val}, check_already_exists=False)

        # Choose the solver for the problem

        solver_key2 = (solver_name2 or solver_name).strip().lower()
        base_solver_kwargs2 = dict(solver_kwargs or {})
        if solver_kwargs2:
            base_solver_kwargs2.update(solver_kwargs2)

        base_solver_kwargs2.setdefault("rtol", 1e-6)
        base_solver_kwargs2.setdefault("atol", 1e-6)
        base_solver_kwargs2.setdefault("root_method", "casadi")

        kw = dict(solver_kwargs)

        if solver_key2 == "idaklu":
            base_solver_kwargs2.setdefault("root_method", base_solver_kwargs2.get("root method", "casadi"))
            solver2 = pybamm.IDAKLUSolver(**base_solver_kwargs2)
        elif solver_key2 == "casadi":
            base_solver_kwargs2.setdefault("mode", base_solver_kwargs2.get("mode", "fast"))
            solver2 = pybamm.CasadiSolver(**base_solver_kwargs2)
        elif solver_key2 == "jax":
            solver2 = pybamm.JaxSolver(**base_solver_kwargs2)
        else:
            raise ValueError(f"Unsupported solver_name {solver_key2}."
                             f"Supported: IDAKLU, Casadi, JAX")

        # Create an option to merge user-supplied model options if they are provided
        model_options2 = dict(model_options)
        if model_options_override2:
            ms2 = (merge_strategy2 or merge_strategy or "override").lower()
            if ms2 == "override":
                # user keys will replace the base keys
                model_options2 = {**model_options2, **model_options_override2}
            elif ms2 == "add_only":
                # add only the new keys
                for k, v in model_options_override2.items():
                    if k not in model_options2:
                        model_options[k] = v
            elif ms2 == "replace":
                model_options = dict(model_options_override2)
            else:
                raise ValueError(f"Unknown merge_strategy '{merge_strategy}'. Use 'override', 'add_only or 'replace'.")

        # Extra constructor kwargs (forwarded)
        stage2_extra_kwargs = dict(extra_model_kwargs or {})
        if extra_model_kwargs2:
            stage2_extra_kwargs.update(extra_model_kwargs2)

        initial_SoC1 = 1
        initial_state2 = {"Initial open-circuit voltage [V]": df2["Voltage [V]"].iloc[0]}
        model2 = model_class(parameter_set=parameterSet, solver=solver2, options=model_options2, **stage2_extra_kwargs)
        model2.set_initial_state(initial_state2)

        # Load the second dataset for stage 2
        if data_file2:
            # Create an absolute path for the data
            dataset_path2 = data_file2

            # importing the experimental/synthetic data
            df2 = pd.read_excel(dataset_path2)
            df2 = df2.iloc[::10].reset_index(drop=True)

            # With the data that we have from the excel file, build the dataset
            dataset2 = pybop.Dataset(
                {
                    "Time [s]": df2["Time [s]"].to_numpy(),
                    "Current function [A]": df2["Current [A]"].to_numpy(),
                    "Voltage [V]": df2["Voltage [V]"].to_numpy()
                }
            )
        else:
            dataset2 = dataset  # reuse stage 1 dataset

            # Assign a signal so PyBOP can see exactly what curve the model is trying to fit when optimising parameters
        signal2 = ["Voltage [V]"]

        stage2_param_dict = {}
        stage2_name_map = {}
        for sk in stage2_keys:
            long_name, default_val = stage2_candidates[sk]
            stage2_param_dict[sk] = default_val
            stage2_name_map[sk] = long_name

        fitting_params2 = []
        bounds_l2 = []
        bounds_u2 = []

        for key in stage2_keys:
            val = stage2_param_dict[key]
            if val <= 0:
                lower, upper = min(0.8 * val, 1.2 * val), max(0.8 * val, 1.2 * val)
            else:
                lower, upper = 0.8 * val, 1.2 * val
            fitting_params2.append(
                pybop.Parameter(
                    stage2_name_map[key],
                    bounds=[lower, upper],
                    initial_value=val
                )
            )
            bounds_l2.append(lower)
            bounds_u2.append(upper)

        lower_bounds2 = np.array(bounds_l2)
        upper_bounds2 = np.array(bounds_u2)
        sigma2 = 0.1 * (upper_bounds2 - lower_bounds2)

        params2 = pybop.Parameters(*fitting_params2)

        x2 = np.array([stage2_param_dict[k] for k in stage2_keys])
        problem2 = pybop.FittingProblem(model=model2, parameters=params2, dataset=dataset2, signal=signal2,
                                        initial_state2=model2.set_initial_state(initial_state))
        cost2 = cost_dict[cost_function](problem2)
        optim2 = optim_dict[optimiser](
            cost2,
            x0=x2,
            verbose=True,
            sigma0=sigma2,
            verbose_print_rate=1,
            allow_infeasible_solutions=False,
            max_iterations=2000,
            max_unchanged_iterations=60,
            parallel=True
        )

        results2 = optim2.run()

        for sk, val_opt in zip(stage2_keys, results2.x):
            lname = stage2_name_map[sk]
            parameterSet[lname] = val_opt
            updated_params_stage2[lname] = val_opt

        '''n_diff_opt, p_diff_opt, a1_opt, a2_opt, a3_opt, n_cond_opt, p_cond_opt, \
        b1_opt, b2_opt, b3_opt = opt_val2'''

        cost_value2 = results2.final_cost
        time2 = results2.time
        iterations2 = results2.n_iterations

        # Printing the results for better visibility
        print(" ")
        print(
            "--------------------------------------------------------------- OPTIMISATION RESULTS ---------------------------------------------------------------")
        print(" ")
        for sk in stage2_keys:
            lname = stage2_name_map[sk]
            init_val = stage2_param_dict[sk]
            print(f"{sk}: {init_val} -> {parameterSet[lname]}   ({lname})")
        print(f"Stage2 final cost: {cost_value2}")
        print(f"Stage2 iterations: {iterations2}")
        print(f"Stage2 time (s): {time2}")
    else:
        print("\n[Stage2] No Stage 2 parameters requested; skipping second optimisation stage.")

        try:
            # Create an initial values list and then evaluate the problem with these values
            init_values2 = [stage2_param_dict[k] for k in stage2_keys]
            pred_init2 = problem.evaluate(init_values2)
            pred_opt2 = problem2.evaluate(results2.x)

            t2 = dataset2["Time [s]"]
            V2 = dataset2["Voltage [V]"]

            pybop.plot.trajectories(
                t2,
                [
                    V2,
                    pred_init2["Voltage [V]"],
                    pred_opt2["Voltage [V]"]
                ],
                trace_names=["Reference Model", "Initial Model", "Optimised Model"],
                xaxis_title="Time / s",
                yaxis_title="Voltage / V"
            )
        except Exception as e:
            print(f"[Stage 2 plot is skipped : {e}]")

    return results, results2


data_file = "/Users/dim/Desktop/PythonFolder/Thesis/Data/ChenDischarge.xlsx"
data_file2 = "/Users/dim/Desktop/PythonFolder/Thesis/Data/ChenDischarge.xlsx"
param_set = "Chen2020"
model = "DFN"
values = ["cn_max", "cp_max", "cn_init", "cp_init", "h", "w", "n_thick", "p_thick", "n_act", "p_act", "n_por", "p_por",
          "n_rad", "p_rad"]
values2 = ["n_diff", "p_diff", "e_diff", "e_cond", "i0_n", "i0_p"]
cost_function = "RMSE"
optimiser = "PSO"

run = pybop_wrapper(
    data_file=data_file,
    data_file2=data_file2,
    param_set=param_set,
    model_selection=model,
    parameters=values,
    solver_kwargs={"rtol": 1e-10, "atol": 1e-10},
    solver_kwargs2={"rtol": 1e-6, "atol": 1e-6},
    parameters2=values2,
    cost_function=cost_function,
    optimiser=optimiser,
    temp=True
)
