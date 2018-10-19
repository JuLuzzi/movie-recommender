from distutils.core import setup

files = ["dataset/*"]

setup(name="movie-recommender",
      version="0.1",
      author="grupo 13",
      author_email="grupo13@utn.frba.com",
      packages=['src'],
      package_data={'src': files},
      scripts=["runner"], requires=['pandas', 'tensorflow', 'matplotlib', 'numpy', 'scikit-learn']
      )
