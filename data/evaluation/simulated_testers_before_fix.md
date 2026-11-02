# Simulated testers

**These are scripted users, not real people.** Their profiles, typed text, hidden dislikes and eating habits were
invented for this test (src/testers.py). The app's own database, planner and feedback functions ran for real on a
throwaway database. No satisfaction ratings exist here, because only real people can give them.

24 testers x 4 weeks x 2 modes (learning on and off) = 192 planned weeks. Assumptions: a meal with a food the
tester dislikes is skipped 90% of the time, other meals are eaten 85-98% of the time, and on a disliked food they swap it
30% of the time and ask for a new meal 15% of the time.

## 1. Learning from behaviour (17 testers with clean wording and a hidden dislike)
Share of meals eaten:

| Mode | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|
| no learning | 78.4% | 80.6% | 77.3% | 79.9% |
| with learning | 78.4% | 88.6% | 90.6% | 87.4% |

Disliked foods served per week (before any swap):

| Mode | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|
| no learning | 5.8 | 5.4 | 5.6 | 5.7 |
| with learning | 5.8 | 1.4 | 0.9 | 0.8 |

## 2. Safety
| Wording | Testers | Allergy understood | Food items checked | Items with a stated allergen | Items breaking a diet rule |
|---|---|---|---|---|---|
| clean | 20 | 100.0% | 12932 | 0 | 0 |
| messy | 4 | 0.0% | 2670 | 185 | 0 |

Messy-wording testers (4): allergy understood by 0; stated allergen served in 185 food items over both modes.

## 3. Accuracy and speed (with learning)
- Calories 2.0%, protein 3.2%, carbs 2.5%, fat 3.8% mean error; 100.0% of days within 10% of the calorie target.
- Plan generation: mean 2.91 s, max 3.82 s.

## 4. Use of the features (both modes)
- Swaps: 148. New meal requests: 79, new version kept: 46.
- Sign-in with the right PIN worked for 100.0% of accounts; a wrong PIN was refused for 100.0%.
- Testers that crashed: 0

## 5. Parser on new wording (61 sentences, written by an AI assistant, not by volunteers)
| Wording | Sentences | Items found | Allergy items blocked |
|---|---|---|---|
| plain | 15 | 17 / 20 | 7 / 9 |
| typos | 10 | 6 / 14 | 3 / 6 |
| indirect | 15 | 6 / 16 | 1 / 5 |
| mixed | 10 | 24 / 25 | 6 / 6 |
| negation | 6 | 2 / 5 | 0 / 0 |
| egyptian | 5 | 4 / 7 | 1 / 1 |
| all | 61 | 59 / 87 | 18 / 27 |

Allergy recall 66.7% (27 items). Recall / precision: never 74.3% / 89.7%,
less often 60.0% / 71.4%, more often 66.7% / 85.7%.
