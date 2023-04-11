## Optimization

delta=np.arange(1.75, 2.75, 0.01)
tau=np.arange(1, 1.5, 0.01)
alpha=np.arange(1.5, 2.5, 0.01)


     for delta in np.arange(1, 3.2, 0.2): #1-3s, 11 pts
        for tau in np.arange(0.75, 1.8, 0.1): #0.75-1.75s, 11pts
            for alpha in np.arange(1.75, 2.30, 0.05): #1.75-2.25s, 11pts

11,11,11 takes about 1s each

| Job Number | Num Delta Pts | Num Tau Pts | Num Alpha Pts | cpus-per-task | Time to run on MLSC (real) |   (user) |   (sys) |
|------------|---------------|-------------|---------------|---------------|----------------------------|----------|---------|
|          0 |             11|           11|             11|             1 |                  7m28.434s | 7m23.144s| 0m0.360s|
|          1 |             11|           11|             11|             2 |                  7m35.671s | 8m40.900s| 6m9.572s|
|          2 |             11|           11|             11|             4 |                   | | |