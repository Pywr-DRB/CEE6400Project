# Working title: Reservoir operational policy inference in the Delaware River Basin

## 0.0 Notation (just to stay consistent ourselves)

The following notation is used:
- $S_t$ is the state space information at time $t$ consisting of:
	- $V_t$ reservoir volume
	- $I_t$ inflow
	- $D_t$ demand
- $\theta$ are the parameters of the control policy
- $\pi(S_t, \theta)$ is a control policy
- $a_t$ is the control action such that $a_t =\pi(S_t, \theta)$ 
- Reservoir characteristics:
	- $V_{max}$ max reservoir volume, capacity
	- $V_{min}$ min reservoir volume, capacity
	- $R_{max}$ max reservoir volume, capacity
	- $R_{min}$ min reservoir volume, capacity
	- $R_{t}$ is the actual release during time $t$


## 1.0 Introduction


## 2.0 Case Study Reservoirs 

Table 1: Data summary for our case study reservoirs


## 3.0 Methodology


### 3.1 Parametric Reservoir Policies

We define a generic operational policy as:
$$\pi (S, \theta)=a$$
Where $S$ is a vector of state space information and $\theta$ is a vector of tunable parameters and $a$ is the control action.

### 3.1.1 Piecewise Linear

$$
a_t = 
\begin{align}
	\left\{
	\begin{array}{lr}
		\theta_{S1} S, & \text{if } & V_t \leq \theta_{V1}\\ 
		\theta_{S2} S, & \text{if } & \theta_{V1} \leq V_t \leq \theta_{V2}\\ 
		\theta_{S3} S, & \text{if } & V_t \geq \theta_{V2}\\ 		 
	\end{array}
	\right\}
\end{align}
$$

### 3.1.2 RBF
Radial basis functions (RBFs) are real-valued functions whose output depends only on the distance from a central point, known as the centroid. The key strength of using RBFs is their ability to approximate complex, nonlinear relationships between the system state and the optimal control action. They are a form of universal approximator meaning that they are flexible enough to describe any function if you combine enough of them. For example, neural networks are also universal approximators.

Gaussian RBFs, are a commonly used variant in the water resources literature (Giuliani et al., 2016; Hamilton et al., 2021). We can define the reservoir policy using a Gaussian RBF as:

$$\pi(S_t) = \sum_{i=1}^m w_i \exp\left(-\frac{||S_t-\mathbf{c}_i||^2}{2\sigma_i^2}\right)$$
Where the parameters of the policy consist of, for each individual RBF, the weights ($w$), the centroids ($c$) and a scale parameter ($\sigma$).

### 3.1.3 STARFIT 

**Need to update this to match other notation above!**

The Normal Operating Range (NOR) is defined as:
$$\dot{S}^{\uparrow} = \mu^{\uparrow}+\alpha^{\uparrow} sin(2\pi\omega t) + \beta^{\uparrow}cos(2\pi\omega t)$$
Then the NOR is defined as:
$$\hat{S}^{\uparrow}_t = 
\begin{align}
	\left\{
	\begin{array}{lr}
		\dot{S^{\uparrow}}, & \text{if } \hat{S^{\uparrow}}_{min} \leq \dot{S^{\uparrow}} \leq \hat{S^{\uparrow}}_{max}\\ 
		\hat{S}^{\uparrow}_{min}, & \text{if } \dot{S^{\uparrow}} \leq \hat{S^{\uparrow}}_{min}\\
		\hat{S}^{\uparrow}_{max}, & \text{if } \hat{S^{\uparrow}}_{max} \leq \dot{S^{\uparrow}} 
	\end{array}
	\right\}
\end{align}$$

The seasonal release harmonic as the sum of two annual harmonics:
$$\tilde{R}_t = \sum_{i=1}^2 \alpha_i cos(2\pi\omega t - \frac{\pi}{2}) + \sum_{i=1}^2 \beta_i cos(2\pi\omega t)$$ The release adjustment
$$\epsilon = c + p_1 A_t + p_2 \hat{I}_t$$
Where $A_t$is the fractional position of the current storage within the NOR (decimal percentage of NOR) and $\hat{I}_t$ is the standardized inflow according to:
$$\hat{I}_t = \frac{I_t - \bar{I}}{\bar{I}}$$

The target release is then:
$$\ddot{R}_t = 
\begin{align}

\left\{
\begin{array}{lr}
	\bar{I} (\tilde{R}_t + \epsilon_t) + \bar{I}, & \text{if } \hat{S}^{\downarrow}_t \leq \hat{S}_t \leq \hat{S}^{\uparrow}_t\\
	S_{cap} (\hat{S}_t - \hat{S}^{\uparrow}_t) + I_t, & \text{if } \hat{S}_t > \hat{S}^{\uparrow}_t \\
	R_{min}, & \text{if } \hat{S}_t < \hat{S}^{\downarrow}_t
	
\end{array}
\right\}
\end{align}$$

The STARFIT formulation thus includes 19 distinct parameters:
- Upper bound of NOR: $\mu^{\uparrow}$, $\alpha^{\uparrow}$, $\beta^{\uparrow}$, $\hat{S}^{\uparrow}_{min}$, $\hat{S}^{\uparrow}_{max}$
- Lower bound of NOR: $\mu^{\downarrow}$, $\alpha^{\downarrow}$, $\beta^{\downarrow}$, $\hat{S}^{\downarrow}_{min}$, $\hat{S}^{\downarrow}_{max}$
- Seasonal release: $\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$
- Release adjustment: $c$, $p_1$, $p_2$
- Release constraints: $R_{min}$, $R_{max}$


For any policy, the actual release is constrained according to the mass balance constraints:
$$R_t = \text{max}(\text{min}(\ddot{R}_t, I_t + S_t), I_t + S_t - S_{cap})$$


### 3.2 Policy Optimization

Describe BorgMOEA and our Borg hyperparameters.
#### 3.2.1 Optimization Objectives

Objective 1: 

Objective 2: 

Objective 3:


### 3.3 Diagnostic Experiment


#### 3.3.i MOEA Convergence
We run N random seeds of the BorgMOEA for each optimization. We compare the hypervolumes generated by each seed to the reference set, representing the overall best parameters across all seeds. 


#### 3.3.i Tradeoff Analysis


#### 3.3.i Policy Performance on Validation Series
After optimizing the parameters, we evaluate the policy performance on a 


## 4.0 Results


## 5.0 Conclusions






## References
- Deisenroth, M. P., Neumann, G., & Peters, J. (2013). A survey on policy search for robotics. Foundations and Trends® in Robotics, 2(1–2), 1-142.
- Giuliani, M., Castelletti, A., Pianosi, F., Mason, E., & Reed, P. M. (2016). Curses, tradeoffs, and scalable management: Advancing evolutionary multiobjective direct policy search to improve water reservoir operations. Journal of Water Resources Planning and Management, 142(2), 04015050.
- Quinn, J. D., Reed, P. M., & Keller, K. (2017). Direct policy search for robust multi-objective management of deeply uncertain socio-ecological tipping points. Environmental modelling & software, 92, 125-141.
- Hadjimichael, A., Reed, P. M., & Quinn, J. D. (2020). Navigating deeply uncertain tradeoffs in harvested predator‐prey systems. Complexity, 2020(1), 4170453.
- Zaniolo, M., Giuliani, M., & Castelletti, A. (2021). Neuro-evolutionary direct policy search for multiobjective optimal control. IEEE Transactions on Neural Networks and Learning Systems, 33(10), 5926-5938.
- Hamilton, A. L., Characklis, G. W., & Reed, P. M. (2022). From Stream Flows to Cash Flows: Leveraging Evolutionary Multi‐Objective Direct Policy Search to Manage Hydrologic Financial Risks. Water Resources Research, 58(1), e2021WR029747.
- Salazar, J. Z., Kwakkel, J. H., & Witvliet, M. (2024). Evaluating the choice of radial basis functions in multiobjective optimal control applications. Environmental Modelling & Software, 171, 105889.
