from distutils.core import setup
setup(
  name = 'broomlib',         # How you named your package folder (MyLib)
  packages = ['broomlib'],   # Chose the same as "name"
  version = '0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library created by Data Science Class of June 2021',   # Give a short description about your library
  author = 'ds-jun21',                   # Type in your name
  author_email = 'gonzalo.perez.diez@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/GonzaloPerez1/broomlib',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/GonzaloPerez1/broomlib/releases/tag/beta_version_01.tar.gz',    # I explain this later on
  keywords = ['data analysis', 'machine learning', 'missings cleaning', 'visualization'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'seaborn',
          'matplotlib',
          'plotly',
          'seaborn',
          'scipy',
          'sklearn',
          'adjustText'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
