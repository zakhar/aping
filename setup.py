from setuptools import setup

setup(
    name='aping',
    version='0.1.0dev',
    py_modules=['aping'],
    url='https://github.com/zakhar/aping',
    license='GPLv3',
    author='Zakhar Zibarov',
    author_email='zakhar.zibarov@gmail.com',
    description='Audio ping',
    install_requires=['numpy', 'sounddevice'],
    entry_points='''
        [console_scripts]
        aping=aping:cli
    '''
)
