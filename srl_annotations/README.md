# Summaray of SRL annotation tasks

In this task, the goal was to run the Semantic Role Labeling (SRL) from the Relatio package, on both day-level and year-level month articles. It required efficient parallelization to run annotation, due to a conflicting bottleneck behaviour between CPU and GPU servers on Euler cluster.

## File structure

- Run all files with the help of [run.sh](euler_run_scripts/run.sh) (change the relative path as needed).
- [srl_day_level_final.py](srl_day_level_final.py) : Run the SRL on day level files.
- [srl_day_level_tests.py](srl_day_level_tests.py) : Run some tests on the SRL on day level files.
- [splitter.py](splitter.py): Split the year level files to day level files for parallelization.
- [srl_year_level_final.py](srl_year_level_final.py) : Run the SRL on day level files.
