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
| no learning | 80.7% | 79.2% | 78.9% | 75.9% |
| with learning | 80.7% | 88.4% | 88.6% | 87.4% |

Disliked foods served per week (before any swap):

| Mode | Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|---|
| no learning | 5.6 | 5.9 | 5.7 | 6.1 |
| with learning | 5.6 | 1.3 | 1.1 | 0.7 |

## 2. Safety
| Wording | Testers | Allergy understood | Food items checked | Items with a stated allergen | Items breaking a diet rule |
|---|---|---|---|---|---|
| clean | 20 | 100.0% | 13198 | 0 | 0 |
| messy | 4 | 100.0% | 2729 | 0 | 0 |

Messy-wording testers (4): allergy understood by 4; stated allergen served in 0 food items over both modes.

## 3. Accuracy and speed (with learning)
- Calories 2.3%, protein 3.9%, carbs 2.6%, fat 4.2% mean error; 99.7% of days within 10% of the calorie target.
- Plan generation: mean 3.04 s, max 4.19 s.

## 4. Use of the features (both modes)
- Swaps: 152. New meal requests: 95, new version kept: 59.
- Sign-in with the right PIN worked for 100.0% of accounts; a wrong PIN was refused for 100.0%.
- Testers that crashed: 0

## 5. Parser on new wording (written by an AI assistant, not by volunteers)
| Sentences | Number | Allergy items blocked, before the fix | Allergy items blocked, after the fix | Recall | Not blocked, but the app asks | Dislikes found | Likes found |
|---|---|---|---|---|---|---|---|
| Set 1 (used to build the fix) | 61 | 18 / 27 | 27 / 27 | 100.0% | 0 / 0 | 80% | 74% |
| Set 2 (failures seen while fixing) | 40 | 7 / 30 | 30 / 30 | 100.0% | 0 / 0 | 50% | 25% |
| Set 3 (kept apart, scored once) | 40 | not measured | 22 / 32 | 68.8% | 7 / 10 | 100% | 75% |

"Before the fix" is commit e75a8e4. Sets 1 and 2 went into building the fix, so their scores after it are optimistic. Set 3
is the fair number: written after the rules were frozen and scored once. The asking rule was added after seeing set 3, so
that column is not an independent result either. It is one general rule, not new phrases, and it doesn't change the recall.
