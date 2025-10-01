# Safe-Explicable-Policy-Search

Welcome to SEPS!

SEPS is built on top of CPO. Specifically, it implements an analytical solution that can solve the constrained optimization problem with two constraints. 
The proof for the analytical solution can be found in the appendices in SEPS.pdf

SEPS is currently implemented in two OpenAI safety gym environments. 
1) pointGoal
2) pointButton; a modified pointGoal env that includes buttons

To run SEPS, run the main program by passing the name of the task. For example,
python main.py --task pointGoal --test --checkpoint checkpoint1

To train in a new environment, follow the following steps
1. Create a new directory with the name of the task
2. Include env.py file which defines the Env class
3. Include the config.yaml file which defines the parameters required for training

For training, simply run
python main.py --task pointGoal

Note:
* SEPS assumes the surrogate reward is available. This is encoded along with the agent's task reward in the Env class


## 📖 Citation

If you use this code in your research, please cite:

@article{hanni2025safe,
  title={Safe Explicable Policy Search},
  author={Hanni, Akkamahadevi and Monta{\~n}o, Jonathan and Zhang, Yu},
  journal={arXiv preprint arXiv:2503.07848},
  year={2025}
}



