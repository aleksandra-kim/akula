from pathlib import Path
import bw2calc as bc
from gsa_framework.utils import read_hdf5_array, write_hdf5_array, write_pickle
from gsa_framework.visualization.plotting import *
from copy import deepcopy
from gsa_framework.models import ModelBase

# COLORS_DICT = {
#     "all": "#636EFA",
#     "influential": "#EF553B",
#     "scatter": "#00CC96",
# }


class LCAModelBW25(ModelBase):

    def __init__(
            self,
            func_unit,
            method,
            write_dir,
            datapackages,
            masks,
            num_params=None,
    ):
        self.lca = bc.LCA(func_unit, method)
        self.lca.lci()
        self.lca.lcia()

        self.write_dir = Path(write_dir)
        self.make_dirs()

        if num_params is None:
            num_params = len(self)
        self.num_params = num_params

        self.datapackages = datapackages
        self.masks = masks

        self.total_num_params = self.get_total_num_params()

    def make_dirs(self):
        """Create subdirectories where intermediate results can be stored."""
        dirs_list = [
            "arrays",
            "figures",
            "LSA_scores",
        ]  # TODO maybe add logging later on
        for dir in dirs_list:
            dir_path = self.write_dir / dir
            dir_path.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return sum([m.sum() for m in self.masks.values()])

    def get_total_num_params(self):

        return 4

    def rescale(self, unit_interval_matrix):
        pass

    def __call__(self, input_parameter_values):
        dict_for_lca = dict(
            use_distributions=False,
            use_arrays=True,
        )
        pass


# class ValidationBW25:
#     """Class to validate sensitivity analysis results with histograms and correlation plots.
#     References
#     ----------
#     Paper:
#         :cite:ts:`kim2021robust`
#     """
#
#     def __init__(
#             self,
#             model,
#             write_dir,
#             iterations=500,
#             seed=None,
#             default_x_rescaled=None,
#             model_output_name="Model output",
#     ):
#         self.model = model
#         self.num_params = len(model)
#         self.write_dir = Path(write_dir)
#         self.make_dirs()
#         self.iterations = iterations
#         self.seed = seed
#         if default_x_rescaled is None:
#             try:
#                 default_x_rescaled = model.default_uncertain_amounts
#             except:
#                 default_x_rescaled = model.rescale(0.5 * np.ones(self.num_params))
#         self.default_x_rescaled = default_x_rescaled
#         self.model_output_name = model_output_name
#         #         self.X_rescaled, self.Y_all = self.generate_X_rescaled_Y_all_parameters_vary()
#         self.X_rescaled = self.generate_X_rescaled_all_inputs_vary()
#         self.Y_all = self.generate_Y_all_inputs_vary()
#
#     def make_dirs(self):
#         """Create subdirectories where intermediate results will be stored."""
#         dirs_list = ["arrays", "figures"]
#         for dir in dirs_list:
#             dir_path = self.write_dir / dir
#             dir_path.mkdir(parents=True, exist_ok=True)
#
#     def generate_X_rescaled_all_inputs_vary(self):
#         """Rescale unitcube samples when all model inputs vary."""
#         if not self.filepath_X_rescaled_all.exists():
#             # Unitcube samples
#             np.random.seed(self.seed)
#             X = np.random.rand(self.iterations, self.num_params)
#             X_rescaled = self.model.rescale(X)
#             write_hdf5_array(X_rescaled, self.filepath_X_rescaled_all)
#         else:
#             X_rescaled = read_hdf5_array(self.filepath_X_rescaled_all)
#         return X_rescaled
#
#     def generate_Y_all_inputs_vary(self):
#         """Run model when all inputs vary."""
#         # Model output
#         if not self.filepath_Y_all.exists():
#             Y = self.model(self.X_rescaled)
#             write_hdf5_array(Y, self.filepath_Y_all)
#         else:
#             # print("{} already exists".format(self.filepath_Y_all.name))
#             Y = read_hdf5_array(self.filepath_Y_all).flatten()
#         return Y
#
#     # def get_fraction_identified_correctly(self, gsa_indices, influential_params_true):
#     #     num_influential = len(influential_params_true)
#     #     influential_params_gsa = np.argsort(gsa_indices)[::-1][:num_influential]
#     #     influential_params_true.sort(), influential_params_gsa.sort()
#     #     non_influential_params_gsa = np.argsort(gsa_indices)[::-1][num_influential:]
#     #     non_influential_params_gsa.sort()
#     #     non_influential_params_true = np.setdiff1d(
#     #         np.arange(self.num_params), influential_params_true
#     #     )
#     #     non_influential_params_true.sort()
#     #     frac_inf = (
#     #         len(np.intersect1d(influential_params_gsa, influential_params_true))
#     #         / num_influential
#     #     )
#     #     frac_non_inf = len(
#     #         np.intersect1d(non_influential_params_gsa, non_influential_params_true)
#     #     ) / len(non_influential_params_true)
#     #     return frac_inf, frac_non_inf
#
#     def get_influential_Y_from_gsa(self, gsa_indices, num_influential, tag=None):
#         """Run model when only influential inputs vary based on sensitivity indices values.
#         Parameters
#         ----------
#         gsa_indices : array
#             Array with sensitivity indices values for all model inputs.
#         num_influential : int
#             Number of first most influential inputs to vary.
#         tag : str
#             Tag to save results.
#         Returns
#         -------
#         influential_Y : array
#             Model outputs when only influential inputs vary.
#         """
#         assert num_influential <= self.num_params
#         assert len(gsa_indices) == self.num_params
#         filepath = self.create_model_output_inf_filepath(num_influential, tag)
#         if filepath.exists():
#             print("{} already exists".format(filepath.name))
#             influential_Y = read_hdf5_array(filepath).flatten()
#         else:
#             non_influential_inds = np.argsort(gsa_indices)[::-1][num_influential:]
#             non_influential_inds.sort()
#             X_rescaled_inf = deepcopy(self.X_rescaled)
#             X_rescaled_inf[:, non_influential_inds] = np.tile(
#                 self.default_x_rescaled[non_influential_inds], (self.iterations, 1)
#             )
#             influential_Y = self.model(X_rescaled_inf)
#             write_hdf5_array(influential_Y, filepath)
#         return influential_Y
#
#     def get_influential_Y_from_parameter_choice(self, influential_inputs, tag=None):
#         """Run model when only influential inputs vary based on chosen influential inputs."""
#         num_influential = len(influential_inputs)
#         assert num_influential <= self.num_params
#         filepath = self.create_model_output_inf_filepath(num_influential, tag)
#         if filepath.exists():
#             print("{} already exists".format(filepath.name))
#             influential_Y = read_hdf5_array(filepath).flatten()
#         else:
#             non_influential_inds = np.setdiff1d(
#                 np.arange(self.num_params), influential_inputs
#             )
#             non_influential_inds.sort()
#             X_rescaled_inf = deepcopy(self.X_rescaled)
#             X_rescaled_inf[:, non_influential_inds] = np.tile(
#                 self.default_x_rescaled[non_influential_inds], (self.iterations, 1)
#             )
#             influential_Y = self.model(X_rescaled_inf)
#             write_hdf5_array(influential_Y, filepath)
#         return influential_Y
#
#     def create_rescaled_samples_all_filename(self):
#         # Maybe we need to be more careful here, as this will change according to the model
#         return "validation.X.rescaled.all.{}.{}.hdf5".format(self.iterations, self.seed)
#
#     def create_model_output_all_filename(self):
#         return "validation.Y.all.{}.{}.hdf5".format(self.iterations, self.seed)
#
#     def create_rescaled_samples_inf_filename(self, num_influential, tag):
#         # Maybe we need to be more careful here, as this will change according to the model
#         filename = "validation.X.rescaled.{}inf.{}.{}.{}.hdf5".format(
#             num_influential, self.iterations, self.seed, tag
#         )
#         filepath = self.write_dir / "arrays" / filename
#         return filepath
#
#     def create_model_output_inf_filepath(self, num_influential, tag):
#         filename = "validation.Y.{}inf.{}.{}.{}.hdf5".format(
#             num_influential,
#             self.iterations,
#             self.seed,
#             tag,
#         )
#         filepath = self.write_dir / "arrays" / filename
#         return filepath
#
#     def create_figure_Y_all_histogram_filepath(self, extension):
#         # Maybe we need to be more careful here, as this will change according to the model
#         filename = "V.histogram.Y.all.{}.{}.{}".format(
#             self.iterations, self.seed, extension
#         )
#         filepath = self.write_dir / "figures" / filename
#         return filepath
#
#     def create_figure_Y_all_Y_inf_histogram_filepath(
#             self, num_influential, tag, extension
#     ):
#         filename = "V.histogram.Y.all.Y.{}inf.{}.{}.{}.{}".format(
#             num_influential, self.iterations, self.seed, tag, extension
#         )
#         filepath = self.write_dir / "figures" / filename
#         return filepath
#
#     def create_figure_Y_all_Y_inf_correlation_filepath(
#             self, num_influential, tag, extension
#     ):
#         filename = "V.correlation.Y.all.Y.{}inf.{}.{}.{}.{}".format(
#             num_influential, self.iterations, self.seed, tag, extension
#         )
#         filepath = self.write_dir / "figures" / filename
#         return filepath
#
#     @property
#     def filepath_X_rescaled_all(self):
#         return self.write_dir / "arrays" / self.create_rescaled_samples_all_filename()
#
#     @property
#     def filepath_Y_all(self):
#         return self.write_dir / "arrays" / self.create_model_output_all_filename()
#
#     def plot_histogram_Y_all(
#             self,
#             default_Y,
#             bin_min=None,
#             bin_max=None,
#             num_bins=60,
#             fig_format=(),
#     ):
#         fig = plot_histogram_Y(
#             Y=self.Y_all,
#             default_Y=default_Y,
#             bin_min=bin_min,
#             bin_max=bin_max,
#             num_bins=num_bins,
#             color=COLORS_DICT["all"],
#             xaxes_title_text=self.model_output_name,
#             trace_name="All parameters vary",
#         )
#         if "pdf" in fig_format:
#             fig.write_image(
#                 self.create_figure_Y_all_histogram_filepath("pdf").as_posix()
#             )
#         if "html" in fig_format:
#             fig.write_html(
#                 self.create_figure_Y_all_histogram_filepath("html").as_posix()
#             )
#         if "pickle" in fig_format:
#             filepath = self.create_figure_Y_all_histogram_filepath("pickle").as_posix()
#             write_pickle(fig, filepath)
#         return fig
#
#     def plot_histogram_Y_all_Y_inf(
#             self,
#             influential_Y,
#             num_influential,
#             tag=None,
#             fig_format=(),
#             bin_min=None,
#             bin_max=None,
#             num_bins=60,
#             showtitle=True,
#     ):
#         fig = plot_histogram_Y1_Y2(
#             self.Y_all,
#             influential_Y,
#             default_Y=None,
#             bin_min=bin_min,
#             bin_max=bin_max,
#             num_bins=num_bins,
#             trace_name1="All parameters vary",
#             trace_name2="Only influential vary",
#             color1="#636EFA",
#             color2="#EF553B",
#             color_default_Y="red",
#             opacity=0.65,
#             xaxes_title_text=self.model_output_name,
#             showtitle=showtitle,
#         )
#         if "pdf" in fig_format:
#             fig.write_image(
#                 self.create_figure_Y_all_Y_inf_histogram_filepath(
#                     num_influential, tag, "pdf"
#                 ).as_posix()
#             )
#         if "html" in fig_format:
#             fig.write_html(
#                 self.create_figure_Y_all_Y_inf_histogram_filepath(
#                     num_influential, tag, "html"
#                 ).as_posix()
#             )
#         if "pickle" in fig_format:
#             filepath = self.create_figure_Y_all_Y_inf_histogram_filepath(
#                 num_influential, tag, "pickle"
#             ).as_posix()
#             write_pickle(fig, filepath)
#         return fig
#
#     def plot_correlation_Y_all_Y_inf(
#             self,
#             influential_Y,
#             num_influential,
#             tag=None,
#             fig_format=(),
#             showtitle=True,
#     ):
#         fig = plot_correlation_Y1_Y2(
#             Y1=self.Y_all,
#             Y2=influential_Y,
#             start=0,
#             end=80,
#             trace_name1="All parameters vary",
#             trace_name2="Only influential vary",
#             yaxes1_title_text=self.model_output_name,
#             xaxes2_title_text=self.model_output_name,
#             yaxes2_title_text=self.model_output_name,
#             showtitle=showtitle,
#         )
#         if "pdf" in fig_format:
#             fig.write_image(
#                 self.create_figure_Y_all_Y_inf_correlation_filepath(
#                     num_influential, tag, "pdf"
#                 ).as_posix()
#             )
#         if "html" in fig_format:
#             fig.write_html(
#                 self.create_figure_Y_all_Y_inf_correlation_filepath(
#                     num_influential, tag, "html"
#                 ).as_posix()
#             )
#         if "pickle" in fig_format:
#             filepath = self.create_figure_Y_all_Y_inf_correlation_filepath(
#                 num_influential, tag, "pickle"
#             ).as_posix()
#             write_pickle(fig, filepath)
#         return fig
