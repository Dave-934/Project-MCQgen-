from setuptools import find_packages,setup
setup(
    name='mcqgenerator',
    version='0.0.1',
    author='Divya Dev',
    author_emails='divya.pandey4@s.amity.edu',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)