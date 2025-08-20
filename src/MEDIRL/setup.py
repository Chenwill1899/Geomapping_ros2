from setuptools import setup

package_name = 'medirl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/medirl.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hydra',
    maintainer_email='hydra@todo.todo',
    description='The medirl package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MEDIRL = medirl.MEDIRL:main',
            'MEDIRLCWL = medirl.MEDIRLCWL:main',
        ],
    },
) 