from setuptools import setup

setup(
    name='VariationalBRAND',
    version='',
    packages=['bin', 'model', 'controller', 'controller.cavi', 'controller.cavi.elbo', 'controller.cavi.utils',
              'controller.cavi.updater', 'controller.cavi.init_cavi', 'controller.specify_user_input',
              'controller.sample_data_handler', 'controller.hyperparameters_setter', 'cavi', 'cavi.elbo',
              'cavi.init_cavi', 'utils', 'parameters_updater', 'specify_user_input', 'sample_data_handler',
              'hyperparameters_setter'],
    package_dir={'': 'root'},
    url='',
    license='',
    author='eb',
    author_email='',
    description=''
)
