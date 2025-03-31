# pycudnnTest

pycudnnTest has two modes: json_graph_test that loads graph definitions from a json file, and python_graph_test that loads graph definitions from a python file

## Quick start

### Graph tests from json
`python pycudnnTest.py --testPath json_graph_defs/graphTests.json --testName ConvRelu1`

### Python defined graph tests
`python pycudnnTest.py --testPath python_graph_defs/basic_tests.py --testName test_conv_relu`