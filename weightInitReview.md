# Feedback

The basic structure and the linguistic quality of the work are good, but there
are some important issues to be addressed (more details in the notes in the
paper):

- it is never clear what the goal of the work is, yes, analysis of output
  patterns, but what do we analyze and why?

- the description of weight init methods is rather confusing, all of them should
  be explained properly in the corresponding section

- figures and tables must be referenced in the text, the captions should be as
  nice as in the result section

- if you use this class/non-class measure, then of course, you have to explain
  this concept, which should be a section in ANNs, because it is never explained
  anyway how classification is performed by the ANN

- you should not make any new experiments, but describe the ones you made
  correctly, and also give some motivation for it, this correlates to the first
  point of explaining what and why you are doing these things

- write a proper summary

# TODO:

## Feedback on page 1

- [ ] Address issues in abstract
- [ ] correctly cite (citation before the dot) in Introduction

## Feedback on page 2

### Section 2.1

- [ ] Remove the "General Information" title
- [ ] Address the "inventions"
- [ ] Correctly cite
- [ ] Address the "summed"
- [ ] Grammar corrections (with -> in, then -> ~then~, some -> a specific)
- [ ] Address the "etc. and"

### Section 2.2

- [ ] Correctly cite
- [ ] Accuraccy and error comment
- [ ] Remove the "and thus learn the same feature"
- [ ] Variance of what?
- [ ] Variance vs gradient, what is the connection, if any?
- [ ] Address the general comment

### Section 2.3

- [ ] Random init "improves" the gradient? Explain
- [ ] symmetry breaking -> symmetry-breaking (?)
- [ ] Explain: much greater results?
- [ ] Remove "close to zero" (?)

## Feedback on page 3

### Section 2.3 (continued)

- [ ] Learning the same feature
- [ ] Comment regarding - "however, the problem is caused by weight init"

### Section 2.4

- [ ] Write Xavier with capital 'X'
- [ ] Number the equation
- [ ] Explain the idea of the formula
- [ ] Correctly explain the Xavier:
  - r is different for each layer
  - Xavier draws from normal, not uniform distribution

### Section 2.5

- [ ] ANN's -> ANN
- [ ] address the comma, maybe change to dot and start a new sentence?
- [ ] altough -> ~altough~
- [ ] map more complex data -> it can make more complex mapping of the data
- [ ] showcase -> show (maybe?)

### Section 2.5.1

- [ ] Binary Step -> maybe rename to Step Function (see comment)
- [ ] First sentence either reword or remove
- [ ] First point is wrong, we need to change that

## Feedback on page 4

### Section 2.5.1 (continued)

- [ ] back propagation -> back-propagation
- [ ] gradient is not a function
- [ ] refer to figure in text (where?)

### Section 2.5.2

- [ ] non-linear -> nonlinear (?)
- [ ] function's gradient -> what to do here?

### Section 2.5.3

- [ ] outputs -> reword this

## Feedback on page 5

### Section 2.5.3 (continued)

- [ ] why does the vanishing gradient occur here?

### Section 2.5.4

- [ ] back propagation -> back-propagation
- [ ] in the ANN at the same time - reword or remove (somehow)
- [ ] back propagation -> back-propagation
- [ ] this issue -> which issue? dead neurons issue?

## Feedback on page 6

### Section 2.5.5

- [ ] As a newly -> remove this?
- [ ] underlying pattenrs -> remove or reword?

### Section 3

- [ ] Experiment setup explanation is vague. Explain more, in particular the
      goal.

### Section 3.1

- [ ] should be integrated in the "Experiment Setup"?
- [ ] database -> dataset
- [ ] similarly designed with precision -> reword
- [ ] in practice -> ~in practice~

## Feedback on page 7

### Section 3.1.1

- [ ] great performance -> reword, what kind of performance
- [ ] was it even trained -> clarify that we did not train the CNN
- [ ] remove the detail about changing shape of the images.

### Section 3.1.2

- [ ] different outputs -> reword this
- [ ] First explain the structure, then variations; see second point for an
      example (?)
- [ ] Output layer point -> last sentence is not clear enough

### Section 3.1.3

- [ ] Used -> ~Used~ (in the title)

## Feedback on page 8

### Section 3.1.3 (continued)

- [ ] Do not explain techniques, refer to previous sections
- [ ] Move (part of) explanation for Xavier to Weight Init Section
- [ ] Random Normal Init -> explained differently here, which one is correct?
- [ ] If we do not use Alternating init, don't describe it. If we do, clarify
      how.

### Section 3.2

- [ ] Explained the metrics too late

### Section 3.2.1

- [ ] Accuracy is not interesting
- [ ] Re-write formula in an obscure way that needs a legen to interpret, like a
      real scientist! 🙄

### Section 3.2.2

- [ ] confidence -> reword/explain
- [ ] Class and Non-Class not explained before, not clear what this means

## Feedback on page 9

### Section 3.2.2 (continued)

- [ ] Re-write formulas, remember to do it like a scientist!
- [ ] No variance of output values, add this

### Section 4

- [ ] Reference section on activation funcions
- [ ] Prof. does not understand the other metric
- [ ] Did we alternate?
- [ ] What about the variance

## Feedback on page 11

- [ ] something on function, don't know what is wanted
- [ ] is low difference good or bad?
- [ ] accuracy is not important, something else?

### Section 5 - Conclusion

- [ ] Rewrite conclusion:
  - [ ] Describe the work and goals; no-one cares about introduction to ANN
  - [ ] Summarize the main results
  - [ ] Discuss results
  - [ ] Give ideas for further work
