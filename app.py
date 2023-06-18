import os
import streamlit as st
import pandas as pd
from tabulate import tabulate
from PIL import Image
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import time
# Import the ExcelHandler class from your Jupyter Notebook
#from single_utils import ExcelHandler, BayesianOptimization, GPModel
from multi_utils import ExcelHandler , MultiObjectiveBO
import datetime
import scipy
from scipy.stats import norm
from simulator import benchmark_fun


# Set page title and favicon
st.set_page_config(
    page_title="SAGE",
    page_icon=":robot_face:",
    layout="wide",
)

col1,empty1,col2 = st.columns([4,8,4])  # creating empty slots

# Display the logo in the last column
# Load logo
logo_image = Image.open("utils/logo_small.png")
#sage_logo = Image.open("utils/sagelogo.png")
col2.image(logo_image,width=180)
#col1.image(sage_logo,width=150)
# Display logo and prompt user to upload Excel file
#st.image(logo_image, width=300)
#st.title("Bayesian Optimization for Smart Experimentation (BOSE)")
#st.subheader('BOSE is an application for performing active experimental design based on Bayesian Optimization.')
tab_about, tab_data, tab_opt, tab_preds, tab_experiment = st.tabs(["ℹ️ About", "🗃 Data ", "📈 Optimization ", "🔍 Visualizations", "🎮 Simulation"])

with tab_about:
    st.markdown(
        """
        ## 💡 What is SAGE?
        Our app, "Smart Adaptive Guidance for Experiments" (SAGE), is a suite for active
        design of experiments. By harnessing the power of the latest advancements in machine learning,
        it solves optimization problems that are complex and expensive to evaluate. Designed to be
        user-friendly and intuitive, it ensures that users, regardless of their technical background,
        can easily use the application to its full potential.
        This is the **multi-objective** module, focused on jointly optimizing two **conflicting** performance metrics. For more capabilities and for
        adapting the software to your specific needs please contact us.

        ## 🖥 How it works?
        The app consists of four main tabs, each with its unique functionality:

        - **🗃 Data**: This is where you upload your initial experimental data. The data should be in an Excel format. Please make
        sure you follow the "template.xslm" file provided and include both the experimental parameters (features) and the observed results (targets).
        - **📈 Optimization**: In this tab, you'll see the optimization process in action. The system will suggest the next set of experiment parameters to try based on the current available data. It uses a Bayesian Optimization approach to intelligently suggest the next experiments.
        The tool provides the flexibility to insert any query that you perform.
        - **🔍 Visualizations**: Here, you can input a set of parameters and the app will give you an estimate of the expected result, along with a probability distribution around that estimate. It allows you to explore the potential outcomes of an experiment
        without having to actually perform it. It also provides further information about the algorithm in terms of sampling utility per point.
        - **🎮 Simulation**: This tab provides a graphical interface for interacting with open-source python modules related to battery modeling
        and simulation. Current version implements a very basic function experiment to serve as a proxy to a real system.


        ## 🚀 Development
        We aim to continuously improve our application by incorporating user feedback, adding new features,
        and adapting to the ever-changing landscape of machine learning and optimization techniques. Please
        contact us for inquiries tailored to your problems.
        """
        )

with tab_data:
    uploaded_file = st.file_uploader("Please upload your Excel file:", type=['xlsm'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # Initialize ExcelHandler object
        excel_handler = ExcelHandler(uploaded_file)
        # Load and display Description sheet
        description_df = excel_handler._read_variable_info()
        st.write("Problem Summary:")
        st.dataframe(description_df)

        # Check if "Observations" sheet is present
        xl = pd.ExcelFile(uploaded_file)
        if "Observations" not in xl.sheet_names:
            st.warning("The 'Observations' sheet does not seem to be present in the uploaded file. This sheet is essential for the Bayesian Optimization process. Please check your file and upload again.")
        else:
            st.success("Data from Excel file has been successfully read")
            bounds_tensor = excel_handler.get_bounds()
            # Get column names from 'Variable Name' values in feature_vars and constraint_vars dataframes
            col_names_x = excel_handler.feature_vars['Variable Name'].tolist()
            # Get the names of all y variables
            target_vars = excel_handler.variable_info[excel_handler.variable_info['Settings'] == 'Target']['Variable Name'].values
            col_names_y = list(target_vars)

        if st.button('Export data'):
            if "df_train" in st.session_state:
                # Filter out the first Ninit rows
                df_export = st.session_state.df_train.iloc[st.session_state.Ninit:]
                # Save to .csv
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # current date and time
                filename = f'study_{timestamp}.csv'
                df_export.to_csv(filename, index=False)
                st.success(f"Data exported to {filename} successfully!")
            else:
                st.warning('No data to export.')



with tab_opt:
    if uploaded_file is not None:
        if "train_x" not in st.session_state or "train_y" not in st.session_state:
            # Need to initialize from excel only when app is loaded
            train_x, train_y = excel_handler.observations_to_tensors()
            st.session_state.Ninit = train_x.shape[0]  # number of initial observations
            # Convert tensors to pandas dataframes
            df_train_x = pd.DataFrame(train_x.numpy(), columns=col_names_x)
            df_train_y = pd.DataFrame(train_y.numpy(), columns=col_names_y)
            # Concatenate train_x and train_y dataframes
            df_train = pd.concat([df_train_x, df_train_y], axis=1)
            # Save the dataframe to the session state
            st.session_state.df_train = df_train
            # Initialize the BO object
            st.session_state.MOBO = MultiObjectiveBO(train_x=train_x, train_y=train_y, bounds=bounds_tensor)
            st.session_state.ref_point = torch.min(train_y, dim=0).values  # note the correction here

        else:
            train_x = st.session_state.train_x
            train_y = st.session_state.train_y
        n_features = len(col_names_x)
        n_targets = len(col_names_y)

        st.markdown("""
            <style>
            .stButton>button {
                background-color: lightgray;
            }
            </style>
            """, unsafe_allow_html=True)

        opt_col1, opt_col2 = st.columns([0.9,1.5])

        with opt_col1:
            # Display the dataframe
            st.markdown("##### Experimental Data")
            st.write("Add completed queries here")
            st.markdown('###')

            # Check if the "Index" column already exists
            if "Index" not in st.session_state.df_train.columns:
                # Generate the index
                index_values = list(range(-st.session_state.Ninit+1, 1)) + list(range(1, len(st.session_state.df_train) - st.session_state.Ninit + 1))
                # Insert the index as the first column
                st.session_state.df_train.insert(0, "Index", index_values)

            # Run the data editor
            st.session_state.df_train = st.experimental_data_editor(st.session_state.df_train, width=450, height=250, num_rows="dynamic")
            # Save any changes back to the session state
            st.session_state.df_train = st.session_state.df_train
            # Find rows without missing values
            complete_rows_mask = st.session_state.df_train.notna().all(axis=1)

            # Separate the target variables (y) from the features (x)
            train_x = st.session_state.df_train[complete_rows_mask].iloc[:, 1:(n_features+1)]  # ignoring "Index" and considering all feature columns
            train_y = st.session_state.df_train[complete_rows_mask].iloc[:, (n_features+1):(n_features+1+n_targets)]  # considering all target columns

            # Convert pandas DataFrames to PyTorch tensors
            train_x = torch.tensor(train_x.values, dtype=torch.float32)
            train_y = torch.tensor(train_y.values, dtype=torch.float32)

            if st.button("Get Experiment"):
                st.markdown("##### Suggested Query")
                st.write("Queue of proposed experiment(s)")
                # Update the BO object
                st.session_state.MOBO = MultiObjectiveBO(train_x=train_x, train_y=train_y, bounds=bounds_tensor, reference_point=st.session_state.ref_point)
                #st.write(train_x.shape)
                #st.write(st.session_state.ref_point.shape)
                suggested_point = st.session_state.MOBO.optimize_acquisition()
                # Convert suggested point to numpy array
                suggested_point_np = suggested_point.detach().numpy().flatten()
                # Construct a DataFrame from suggested point with appropriate column names
                st.session_state.suggested_df = pd.DataFrame([suggested_point_np], columns=col_names_x)
                # Add "Value" as column header
                st.session_state.suggested_df = st.session_state.suggested_df.rename_axis("Value", axis=1)

            # Display the stored suggested_df if it exists
            if hasattr(st.session_state, 'suggested_df'):
                table = tabulate(st.session_state.suggested_df, tablefmt="pipe", headers="keys", showindex=False)
                st.write(table, unsafe_allow_html=True)

        with opt_col2:
            st.markdown("##### Performance Progress")
            st.write("A visualization of the algorithm progression")
            df_train = st.session_state.df_train
            n_targets = len(col_names_y)

            # Initialize MOBO in the session state if it doesn't exist
            if 'MOBO' not in st.session_state:
                st.session_state.MOBO = MultiObjectiveBO(train_x=train_x[:st.session_state.Ninit], train_y=train_y[:st.session_state.Ninit, :], bounds=bounds_tensor)

            # Get initial mean and standard deviation for all targets
            mean_init, std_dev_init = st.session_state.MOBO.get_posterior_stats(train_x[:st.session_state.Ninit])

            # Create a subplot with n_target rows
            fig = make_subplots(rows=n_targets, cols=1, subplot_titles=col_names_y, vertical_spacing=0.15)

            x_axis = list(range(-st.session_state.Ninit+1, 1)) + list(range(1, df_train.shape[0]-st.session_state.Ninit+1))

            # Add the initial training data to the plot
            for target_idx in range(n_targets):
                fig.add_trace(
                    go.Scatter(
                        x=x_axis[:st.session_state.Ninit],
                        y=train_y[:st.session_state.Ninit, target_idx],
                        mode='markers',
                        marker=dict(color='lightblue', symbol='circle', size=20),
                        name='Initial Observed Performance',
                        legendgroup="group1",
                        hovertemplate="Initial Observed Performance",
                        showlegend= target_idx == 0
                    ),
                    row=target_idx+1,
                    col=1
                )

            for i in range(st.session_state.Ninit, len(df_train)):
                for target_idx in range(n_targets):
                    # Separate the target variable (y) for the subplot
                    df_train_y = df_train.iloc[:, -(n_targets-target_idx)]  # assuming y's are the last columns

                    # Check if the actual observation is available
                    if not np.isnan(df_train_y[i]):
                        if not np.isnan(df_train_y[i]):
                            # Update the BO object in the session state
                            st.session_state.MOBO = None
                            if 'MOBO' not in st.session_state:
                                st.session_state.MOBO = MultiObjectiveBO(train_x=train_x[:i], train_y=train_y[:i,:], bounds=bounds_tensor)
                                mean, std_dev = st.session_state.MOBO.get_posterior_stats(train_x[:i+1])
                            else:
                                st.session_state.MOBO = MultiObjectiveBO(train_x=train_x[:i], train_y=train_y[:i,:], bounds=bounds_tensor)
                                mean, std_dev = st.session_state.MOBO.get_posterior_stats(train_x[:i+1])

                        #mean, std_dev = st.session_state.MOBO.get_posterior_stats(train_x[:i+1])

                        # Add trace for predicted mean with error bars
                        fig.add_trace(
                            go.Scatter(
                                x=x_axis[i:i+1],
                                y=mean[-1:, target_idx].detach().numpy().flatten(),
                                error_y=dict(
                                    type='data',
                                    array=1.96*std_dev[-1:, target_idx].detach().numpy().flatten(),
                                    visible=True
                                ),
                                mode='markers',
                                marker=dict(color='mediumslateblue', size=20),
                                name='Predicted Performance',
                                legendgroup="group2",
                                hovertemplate="Predicted Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                                showlegend= i == st.session_state.Ninit and target_idx == 0
                            ),
                            row=target_idx+1,
                            col=1
                        )

                        # Add trace for actual observation
                        fig.add_trace(
                            go.Scatter(
                                x=x_axis[i:i+1],
                                y=df_train_y[i:i+1],
                                mode='markers',
                                marker=dict(color='mediumvioletred', symbol='star', size=20),
                                name='Observed Performance',
                                legendgroup="group3",
                                hovertemplate="Observed Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                                showlegend= i == st.session_state.Ninit and target_idx == 0
                            ),
                            row=target_idx+1,
                            col=1
                        )

            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=15)),
                shapes=[
                    dict(
                        type="line",
                        yref="paper", y0=0, y1=1,
                        xref="x", x0=0, x1=0,
                        line=dict(
                            color="Black",
                            width=1,
                            dash="dot",
                        )
                    )
                ],
                height=800,  # Adjust the height of the plot
                margin=dict(b=150)  # Increase the bottom margin
            )

            for i in range(n_targets):
                fig.update_yaxes(title_text="Value", row=i+1, col=1)
            fig.update_xaxes(title_text="Iteration", row=n_targets, col=1)

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("No file has been uploaded yet.")


# Define color list
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

with tab_preds:
    if uploaded_file is not None:
        preds_col1, preds_col2 = st.columns([0.7,1.0])

        with preds_col1:
            st.markdown("##### Inspection at Desired Inputs")
            st.write("Add test points at which you want to <br> estimate the performance", unsafe_allow_html=True)
            st.write("#")
            st.write("#")
            if "df_test" not in st.session_state or st.session_state.df_test is None:
                max_index = train_y.argmax().item()-1
                max_input = train_x[max_index]
                initial_test_df = pd.DataFrame(max_input.numpy()[np.newaxis, :], columns=col_names_x)
                st.session_state.df_test = initial_test_df.reset_index().rename(columns={'index': 'Index'})

            df_test = st.experimental_data_editor(st.session_state.df_test, width=500, height=200, num_rows="dynamic")
            st.session_state.df_test.reset_index(drop=True, inplace=True)
            # Find rows without missing values

            # Initialize the figure outside of the loop
            fig = go.Figure()

            # Plot the training data
            complete_rows_mask_train = st.session_state.df_train.notna().all(axis=1)
            train_y = st.session_state.df_train[complete_rows_mask_train].iloc[:, (n_features+1):(n_features+1+n_targets)]  # considering all target columns
            train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32)
            pareto_y = st.session_state.MOBO.get_pareto_points(train_y_tensor)

            # Add shaded rectangles to indicate dominated region
            x_pareto = pareto_y[:, 0].cpu().numpy()
            y_pareto = pareto_y[:, 1].cpu().numpy()
            max_y = max(np.max(y_pareto), 0) # Use the maximum y-value for filling

            #x_fill = np.concatenate(([x_pareto[0]], x_pareto, [x_pareto[-1]]))
            #y_fill = np.concatenate(([np.max(y_pareto)], np.maximum.accumulate(y_pareto), [0]))
            #fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='tozeroy', mode='none', fillcolor='rgba(192,192,192,0.5)', showlegend=False))
            # Sort the Pareto points by the first objective
            pareto_y_sorted = pareto_y[pareto_y[:, 0].argsort()]

            # Add lines connecting adjacent Pareto points and fill the area under each line
            for i in range(pareto_y_sorted.shape[0] - 1):
                x_fill = [pareto_y_sorted[i, 0], pareto_y_sorted[i+1, 0], pareto_y_sorted[i+1, 0], pareto_y_sorted[i, 0]]
                y_fill = [0, 0, pareto_y_sorted[i+1, 1], pareto_y_sorted[i, 1]]
                fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='toself', fillcolor='rgba(192,192,192,0.5)', line=dict(width=0), mode='lines', showlegend=False, hovertemplate='Dominated Region'))


            fig.add_trace(go.Scatter(x=train_y_tensor[:, 0].cpu().numpy(),
                                     y=train_y_tensor[:, 1].cpu().numpy(),
                                     mode='markers',
                                     marker=dict(size=10, color='black'),
                                     name='Training data'))

            # Get and plot the Pareto optimal points
            fig.add_trace(go.Scatter(x=pareto_y[:, 0].cpu().numpy(),
                                     y=pareto_y[:, 1].cpu().numpy(),
                                     mode='markers',
                                     marker=dict(size=10,color='black'),
                                     name='Pareto Optimal Points',
                                     showlegend=False))

            st.session_state.plot = fig

            # Check if all rows are complete
            #if complete_rows_mask_test.all():


            if st.button("Predict"):
                st.session_state.df_test = df_test  # Store user input
                complete_rows_mask_test = st.session_state.df_test.notna().all(axis=1)

                test_x = st.session_state.df_test[complete_rows_mask_test].iloc[:, 1:]  # Exclude 'Point Index'
                # Convert pandas DataFrames to PyTorch tensors
                test_x = torch.tensor(test_x.values, dtype=torch.float32)
                # Initialize lists for means and variances
                means = []
                std_devs = []

                # Make predictions for complete rows
                for i, idx in enumerate(test_x):
                    input_tensor = torch.tensor(idx).reshape(1, -1)
                    mean, std_dev = st.session_state.MOBO.get_posterior_stats(input_tensor)

                    means.append(mean)
                    std_devs.append(std_dev)

                    # Calculate the ellipse points for each mean and std_dev pair
                    # Number of points to generate
                    num_points = 300
                    # The t values to generate points
                    t = np.linspace(0, 2*np.pi, num_points)

                    # Generate the points of the ellipse
                    x = mean[0, 0].cpu().numpy() + 1.96* std_dev[0, 0].cpu().numpy() * np.cos(t)
                    y = mean[0, 1].cpu().numpy() + 1.96 * std_dev[0, 1].cpu().numpy() * np.sin(t)

                    # Add ellipse points to plot
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='toself',
                                             line=dict(color=color_list[i % len(color_list)]),
                                             name=f"Point {i}"))

                    # Add the center of the ellipse to the plot
                    # Add the center of the ellipse to the plot
                    fig.add_trace(go.Scatter(x=[mean[0, 0].cpu().numpy()],
                                             y=[mean[0, 1].cpu().numpy()],
                                             mode='markers',
                                             marker=dict(size=10, color=color_list[i % len(color_list)], line=dict(color='black', width=2)),
                                             showlegend=False,
                                             name=f"Center {i}"))


                # Update axes labels
                fig.update_xaxes(title_text=col_names_y[0])
                fig.update_yaxes(title_text=col_names_y[1])

                st.session_state.plot = fig
                st.session_state.means = means
                st.session_state.std_devs=std_devs
            #else:
            #    st.error("Please fill in all fields before predicting.")

        with preds_col2:
            st.markdown("##### Visualization of Objective Space")
            st.write("Estimated performance statistics based on <br> available observations", unsafe_allow_html=True)
            if 'plot' in st.session_state:
                st.write(st.session_state.plot)



        st.markdown("##### Optimization Progress")
        st.write("""This graphic illustrates the progression of
        the optimization process over several iterations.
        In the context of multi-objective optimization, we monitor the hypervolume (HV).
        The HV signifies the volume of the area that is dominated by the points we've observed.
        A larger dominated region is desirable, as it indicates that we've identified points that
        best balance the different objectives, a concept known as Pareto optimality.""", unsafe_allow_html=True)

        complete_rows_mask_train = st.session_state.df_train.notna().all(axis=1)
        train_df = st.session_state.df_train[complete_rows_mask_train]
        train_y = train_df.iloc[:, (n_features+1):(n_features+1+n_targets)]  # considering all target columns
        train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32)

        hv_values = []
        index_values = train_df['Index'].values.tolist()

        for i in range(1, train_y_tensor.shape[0] + 1):
            hv = st.session_state.MOBO.compute_hypervolume(train_y_tensor[:i])
            hv_values.append(hv)

        df_hypervolume = pd.DataFrame({
            'Index': index_values,
            'Hypervolume': hv_values
        })

        fig_hypervolume = px.line(df_hypervolume, x='Index', y='Hypervolume', template='plotly_white', markers=True)
        fig_hypervolume.update_layout(autosize=False, width=600, height=350)
        st.write(fig_hypervolume)




    else:
       st.write("No file has been uploaded yet")



with tab_experiment:
    st.write("### Experiment Setup")
    st.write("""Run simulated experiment from high-fidelity simulator and get
    objective values of interest. This is a example is meant as a demonstration of the SAGE
    capabilities and use.""")

    # create two columns
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.write("#### User Inputs")
        st.write("""Edit the cells below and add the desired input values at which you want to perform the
        experiment at.""")

        x1 = st.number_input("Parameter 1", value=0.5, step=0.01, format="%.2f")
        x2 = st.number_input("Parameter 2", value=0.5, step=0.01, format="%.2f")
        run_button = st.button("Run Experiment")

    with exp_col2:
        st.write("#### Experiment Outcome")
        st.write("""This experiment runs the benchmark function that calculates two metrics. The experiment
        is widely used for validation of optimization methods. """)

        if run_button:
            # Create a numpy array with the input values
            input_values = np.array([x1, x2])

            # Call the benchmark function with the input values
            f_values = benchmark_fun(input_values)

            # Prepare the output as a DataFrame
            output_df = pd.DataFrame({
                'Metric Name': ['System Performance 1', 'System Performance 2'],
                'Value': f_values
            })

            # Convert the numbers to strings with 3 decimal places
            output_df['Value'] = output_df['Value'].apply(lambda x: "%.3f" % x)

            # Convert the DataFrame to HTML and apply CSS styling
            output_html = output_df.style.set_table_styles(
                [{
                    'selector': 'th',
                    'props': [('background', 'rgba(173, 216, 230, 0.5)'),  # light blue background
                              ('color', 'black'), ('font-size', '90%')
                              ]
                },
                {
                        'selector': 'td',
                        'props': [
                            ('font-size', '90%')  # decrease font size
                        ]
                    }

                ]
            ).hide_index().render()

            # Place the objective values table in the right subcolumn
            st.markdown(output_html, unsafe_allow_html=True)
