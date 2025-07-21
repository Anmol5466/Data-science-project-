# Install PuLP if not already installed
# !pip install pulp

from pulp import LpMaximize, LpProblem, LpVariable, value

# Step 1: Define the problem
model = LpProblem("Product_Mix_Optimization", LpMaximize)

# Step 2: Define decision variables
A = LpVariable("Product_A", lowBound=0, cat='Integer')
B = LpVariable("Product_B", lowBound=0, cat='Integer')

# Step 3: Define objective function (maximize profit)
model += 30 * A + 50 * B, "Total_Profit"

# Step 4: Define constraints
model += 3 * A + 4 * B <= 240, "Machine_1_Limit"
model += 2 * A + 3 * B <= 180, "Machine_2_Limit"

# Step 5: Solve the problem
model.solve()

# Step 6: Display results
print(f"Status: {model.status}, {LpProblem.status[model.status]}")
print(f"Produce {A.varValue} units of Product A")
print(f"Produce {B.varValue} units of Product B")
print(f"Maximum Profit: ₹{value(model.objective)}")