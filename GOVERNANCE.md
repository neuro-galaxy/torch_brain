# TorchBrain Governance Charter

The purpose of this document is to formalize the governance process of the
project, to clarify how decisions are made and how the various members of the
community interact.

**Summary of project roles:**

1. **Contributors:** contribute code or other artifacts, but do not have
   write-access to the source repository.
2. **Core-Contributors:** make sustained contributions and have write-access to
   the source repository.
3. **The Technical Committee (TC):** core-contributors appointed to oversee the
   technical development of the project.
4. **The Advisory Committee (AC):** senior researchers appointed to oversee the
   project’s mission and set long-term objectives.

## §1 Decision making process

The TC uses a [consensus seeking](https://en.wikipedia.org/wiki/Consensus_decision-making)
process for all its decisions. The group tries to find a resolution that has no
objection among TC members. If a consensus cannot be reached then a majority
vote is called. It is expected that the TC makes decisions primarily via a
consensus seeking process and that voting is only used as a last-resort.

When an agenda item has appeared to reach a consensus the TC moderator (§3)
will ask “Does anyone object?” as a final call for dissent from the consensus.
All decisions are expected to reach consensus within two TC meetings. Decisions
that have not reached a consensus for two or more meetings should automatically
be called to a vote, unless there is consensus for elongating the discussion of
the issue for one more meeting. A majority of TC members are required to be
present in the TC meeting when making any consensus-based decisions.

In the event of a tie vote, the proposal or motion does not move forward. The
proposer may revise and resubmit the motion for a subsequent meeting.

## §2 Responsibilities of the TC

TC is responsible for and has decision authority for all technical development
of the project, including:

- Technical direction.
- Setting release dates and signing-off on releases.
- Development process and any coding standards.
- Managing core infrastructure (e.g. GitHub, PyPI, websites, discussion channels).
- Conduct guidelines (A code of conduct is TBD, potentially [Contributor Covenant](https://www.contributor-covenant.org/)).
- Mediating technical conflicts between Core-Contributors.
- Maintaining and publishing the list of current Governance members.

## §3 Establishment of the TC

Core-Contributors who have demonstrated both broad technical familiarity with
the project and sound judgment in technical decisions are considered for TC
membership. A motion to add TC members can be put forward by any TC/AC member
and its resolution must follow §1.

TC members are expected to regularly participate in TC activities and uphold
the “Decision making process” guidelines (§1). The TC shall meet at least once
every month to deliberate on any issues and proposals that have not reached
consensus. Meetings are moderated by the **TC moderator**, a rotating role held
by a TC member for a term of 3 months, after which it passes to another TC
member. The TC moderator may delegate responsibility for directing an
individual meeting to any other TC member. The TC moderator shall ensure that
meeting notes are taken and made accessible to the TC, AC, and
Core-Contributors. Any Core-Contributor or TC/AC member may add items to the
agenda of the TC meeting.

The TC must have at least three members. In the case that the TC member count
drops below three, an open election among all Core-Contributors must be held to
elect another member.

Admin access to core infrastructure (e.g. GitHub, PyPI, websites, discussion
channels) must be held by at least three TC members at all times.

A motion to remove a TC member can be put forward by any TC/AC member and its
resolution must follow §1. The affected person should be notified and given an
opportunity to respond before the vote. The affected person may not vote on
their own removal. In case of a tie vote in this matter, the AC is called to
vote, and the combined TC + AC tally determines the outcome. In case a TC/AC
member may hold a position of direct professional authority over the affected
member, they must also recuse themselves from voting.

## §4 Core-Contributors

Individuals making sustained contributions are made **Core-Contributors** by
the TC on an ongoing basis. Sustained contributions are evaluated holistically
by the TC and may include code, documentation, code review, community support,
or infrastructure work. These individuals are identified by the TC/AC and their
addition as Core-Contributors is a TC decision.

Core-Contributors have write-access to the source repository and the right to
approve and merge PRs that have reached a [Lazy Consensus](https://openoffice.apache.org/docs/governance/lazyConsensus.html#:~:text=Lazy%20consensus%20means%20we%20can,happening%2C%20as%20it%20is%20happening.),
that is, when it has no public objections. Trivial changes (typo/bug fixes, CI
configuration, documentation corrections, small features) may be merged with
one approval and no mandatory waiting period. For non-trivial PRs (marked by
the `non-trivial` tag), lazy consensus requires a minimum review period of 72
hours and at least two approvals from Core-Contributors before merging. The
author of the PR may not count as an approving reviewer.

Core-Contributors may opt to elevate significant or controversial
modifications, or modifications that have not found consensus, to the TC for
discussion by assigning the `tc-agenda` tag to a pull request or issue. All
Contributors can raise an issue to be discussed by the TC through a GitHub
issue with the `tc-agenda` label in the project's primary repository.

A motion to remove a Core-Contributor can be brought forth to the TC by other
Core-Contributors or TC/AC members. The affected person should be notified in
advance and given an opportunity to respond before the TC deliberates and
votes.

## §5 Establishment of the AC

The AC consists of senior members of the research community that set long-term
objectives and serve as liaisons for engaging with the wider community of
researchers.

AC members may attend TC meetings and add items to be discussed in the TC
meeting agenda. A joint AC + TC meeting shall be held once every quarter to
discuss progress and set goals for the future. Any TC/AC member may also call a
joint meeting to seek guidance and discuss issues that cannot be resolved
solely by the TC.

**Independence of the TC.** The TC makes technical decisions independently and
in the best interest of the project. Professional or institutional
relationships that exist between AC and TC members outside the project are
considered external to the charter and should not influence TC deliberations or
decisions.

## §6 Amendments to this charter

Amendments to this charter may be proposed by any TC/AC member and the decision
is made by a joint vote of the TC and AC, following the consensus-seeking
process of §1 applied to the combined body.

## §7 General governance logistics

**Elections:** All elections must follow the [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method)
(e.g. using [CIVS](https://civs1.civs.us/)).

**Removal process votes:** All votes regarding removing members must be
conducted anonymously. Anonymous votes must be conducted online and allow for a
voting window of 72 hours.

**Asynchronous decisions:** All TC or AC consensus seeking processes can be
performed through email discussion. If the asynchronous consensus-seeking
process elongates for more than two weeks, an online vote is automatically
called, unless there is consensus for elongation of discussion.

## §8 Current governance structure

**Advisory Committee members** (last name order):

- Dr. Mehdi Azabou [mehdi [at] constellationlab [dot] io]
- Dr. Eva L. Dyer [eva [dot] dyer [at] seas [dot] upenn [dot] edu ]
- Dr. Blake A. Richards [blake [dot] richards [at] mila [dot] quebec]

**Technical Committee members** (last name order):

- Alexandre Andre [aandre1 [at] seas [dot] upenn [dot] edu]
- Vinam Arora (current moderator) [vinam [at] seas [dot] upenn [dot] edu]
- Milo Sobral [milo [dot] sobral [at] mila [dot] quebec]

