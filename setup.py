from distutils.core import setup

setup(
    name='DreamEngine',
    version='0.0.1',
    packages=['', 'dreamengine'],
    url='www.dreamengine.com',
    license='proprietary DO NOT STEAL',
    author='chocolate outline',
    author_email='alcornwill@gmail.com',
    description='pure python game engine',
    install_requires=[
        'pygame',
        'numpy',
        'pyopengl',
        'pyassimp'
    ]
)

