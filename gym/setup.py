from setuptools import find_packages, setup

setup(
    name="sftvsrl-gym",
    version="0.0.2",
    packages=find_packages(),
    package_data={'gym_cards': ['envs/img/*.png', 'envs/font/*.ttf']},
    install_requires=["gymnasium", "numpy", "Pillow"]
)