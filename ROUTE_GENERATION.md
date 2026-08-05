## why generate routes
55 stores need pallets every day from the mt roskill warehouse, trucks hold 16 pallets. solving that all at once is too much, so we do it in two stages. first build a pool of legal routes, then a milp picks the cheapest set that covers every store. this file is stage one, the milp is the next step.
## sorting out the data
we have a 56x56 drive time matrix for the 55 stores plus the warehouse, estimated demand per store per day, and 18 min unload per pallet. early bug was the warehouse getting treated like a store, so indices pointed at the wrong columns and every route was priced off wrong times. fixed by reindexing the warehouse to the last position, index 55. matrix math checked out after that.
## the cost model
- 220/hr for the first 3.5 hours
- 310/hr overtime after that
- plus 18 min unload per pallet
we used a soft cap with overtime instead of a hard one, because a hard cap throws away routes that are slightly over the line and youd pay the overtime anyway. kept a 6 hour cap on multi stops, single stops are exempt.
## v1 the first attempt
heuristic generator, savings approach plus random insertion. covered every store every day but the stop order was random and the pool only had routes the heuristic happened to stumble on.
## v2 fixing the ordering
cost is almost entirely decided by the stop order, so we added a 2 opt plus relocate pass and reordered every route. kept the same store sets so the pools are comparable. many came back cheaper, drive time down a couple percent, cost down around 1 percent. worth doing.
## v3 doing it properly
the heuristic was still leaving good routes out, so we stopped guessing
- enumerate every feasible route with 1 to 4 stops, so every good small route is guaranteed in the pool
- kill routes that can never be picked, like a 2 stop trip that costs more than running two singles, no optimal solution would pick it. dominance pruning, removed a lot of v2s routes
- 2 opt everything that survives
- add 5 to 6 stop routes via savings and random where enumeration is too expensive
saturday had too many routes, so we capped the pool at 40000 cheapest non dominated routes a day. coverage still 100 percent.
there was also a bug in the v3 rewrite, an active store flag was set wrong and some stores dropped out of the pool. fixed and re checked coverage.
## where its at now
v3 is the pool the milp will solve over, complete through 4 stops on weekdays, 2 opt ordered, capped on saturday. verified capacity, coverage, drive times. v3 routes arent individually cheaper than v2s, the value is a better pool with fewer useless routes and more good multi stop options. the actual savings show up when the selection step picks the combination.
