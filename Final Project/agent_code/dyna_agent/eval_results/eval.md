The original schema comparison results were lost through migrating computers partway through the project, 
so the schema_comparison_table is a recreation of the table in the existing report. 
The regenerated results lie within the existing confidence intervals. The training
curve csv is original.

Wins counted in schema_comparison mean only higher than all other opponents with ties
counting as losses. The rule-based column is an average of the three rule-based opponents
each round before taking the mean, the round is the unit of independence. For the 
schema table and training graph, 400 rounds were run on seed 123, while the other 
400 were run on seed 7. The DynaQ agent was run against three rule-based agents
in both evaluations. The environment.py file was edited to include per round 
and per agent statistics. 