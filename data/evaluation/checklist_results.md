# Allergy checklist

## Plans for people who tick and type nothing (15 people, 1 week each)
- Food items served: 1239. Items containing a ticked allergen: **0**.
- Mean calorie error: 1.9%.

## Sentence sets, if the person also ticked the matching box
| Sentences | Allergy items blocked, text alone | With the box ticked | Named foods the checklist doesn't list, still missed |
|---|---|---|---|
| Set 1 (used to build the fix) | 27 / 27 | 27 / 27 | 0 |
| Set 2 (failures seen while fixing) | 30 / 30 | 30 / 30 | 0 |
| Set 3 (kept apart, scored once) | 22 / 32 | 32 / 32 | 1 |

The "with the box ticked" column is true by design: a ticked group is always blocked, whatever is typed. It assumes that people tick the
right box, which has not been tested with real people. Allergies outside the 10 listed groups, such as kiwi, still depend on the free text.
