(define (domain quadcopter)
 (:requirements :strips :typing)

 (:types
    quadcopter location
 )

 (:predicates
    (at ?q - quadcopter ?l - location)
    (adjacent ?l1 - location ?l2 - location)
    (target ?l - location)
 )

 (:action move
  :parameters (?q - quadcopter ?from - location ?to - location)
  :precondition (and
        (at ?q ?from)
        (adjacent ?from ?to)
  )
  :effect (and
        (not (at ?q ?from))
        (at ?q ?to)
  )
 )

 (:action reach_target
  :parameters (?q - quadcopter ?l - location)
  :precondition (and
        (at ?q ?l)
        (target ?l)
  )
  :effect (reached ?q)
 )
)