# Contributing to Cities Skylines 2 Agent

Thank you for considering contributing to the Cities Skylines 2 Agent project! This document outlines the process for contributing to the project and provides guidelines for code style, testing, and submission.

## Getting Started

1. **Fork the repository**: Start by forking the repository to your GitHub account.

2. **Clone your fork**: Clone your fork to your local machine.
   ```bash
   git clone https://github.com/yourusername/cities-skylines-2-agent.git
   cd cities-skylines-2-agent
   ```

3. **Create a virtual environment**: Set up a virtual environment to work in.
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies**: Install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a branch**: Create a branch for your changes.
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

This project follows these code style guidelines:

- **PEP 8**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines for Python code.
- **Docstrings**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for documentation.
- **Type Hints**: Use type hints for function arguments and return values.
- **Imports**: Organize imports in three groups: standard library, third-party, and local modules.
- **Line Length**: Limit line length to 88 characters.

You can check code style using tools like `flake8`, `black`, and `isort`:

```bash
# Check code style
flake8 src tests

# Format code
black src tests

# Sort imports
isort src tests
```

## Project Structure

The project follows this structure:

```
cities-skylines-2/
├── src/                # Source code
│   ├── agent/          # Agent components
│   ├── environment/    # Environment components
│   ├── model/          # Neural network models
│   ├── training/       # Training infrastructure
│   ├── utils/          # Utility modules
│   └── config/         # Configuration
├── docs/               # Documentation
├── tests/              # Tests
├── scripts/            # Utility scripts
└── logs/               # Log files
```

When adding new files, follow these guidelines:

- Place new modules in the appropriate directory based on functionality.
- Create appropriate `__init__.py` files to expose the module interface.
- Keep modules focused on a single responsibility.
- Document modules with clear docstrings.

## Testing

All contributions should include appropriate tests to verify functionality:

1. **Write tests**: Add tests in the `tests` directory for your changes.
2. **Run tests**: Ensure all tests pass before submitting.
   ```bash
   python -m unittest discover tests
   ```

3. **Coverage**: Aim for good test coverage of new functionality.

## Submitting Changes

1. **Commit your changes**: Make a commit with a descriptive message.
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

2. **Push your branch**: Push your branch to your fork.
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request**: Create a pull request from your fork to the original repository.

In your pull request description:
- Describe what your changes do
- Reference any related issues
- Mention any particular points reviewers should focus on

## Pull Request Review Process

Once you submit a pull request:

1. Maintainers will review your code
2. Automated tests will run to verify your changes
3. You may be asked to make changes based on the review
4. Once approved, your changes will be merged

## Development Environment

For the best development experience:

1. Use an IDE with Python support (Visual Studio Code, PyCharm, etc.)
2. Install recommended extensions for your IDE:
   - Python language support
   - Linting tools (flake8, pylint)
   - Formatting tools (black)
   - Import sorting (isort)

## Working with Issues

- Check existing issues before creating new ones
- Use issue templates when available
- Tag issues appropriately (bug, enhancement, etc.)
- When fixing an issue, reference it in your commit message and pull request

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have any questions about contributing, feel free to create an issue asking for guidance. 