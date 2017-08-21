import os
from distutils.core import setup

package_name = "pyrats"
version = '0.1'
README = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = open(README).read()
setup(name=package_name,
      version=version,
      description=("Tree and Halo manager"),
      long_description=long_description,
      classifiers=[
          "Programming Language :: Python",
          ("Topic :: Software Development :: Libraries :: Python Modules"),
      ],
      keywords='data',
      author='Maxime Trebitsch, Hugo Pfister',
      author_email='maxime.trebitsch@iap.fr, hugo.pfister@iap.fr',
      license='MIT',
      package_dir={package_name: package_name},
      packages=[package_name],
      install_requires=["yt", "pandas"]
)
