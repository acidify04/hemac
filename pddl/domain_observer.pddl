(define (domain observer)
 (:requirements :strips :typing)

 (:types
    observer location
 )

 (:predicates
    (at ?o - observer ?l - location)
    (adjacent ?l1 - location ?l2 - location)
 )

 (:action move
  :parameters (?o - observer ?from - location ?to - location)
  :precondition (and
        (at ?o ?from)
        (adjacent ?from ?to)
  )
  :effect (and
        (not (at ?o ?from))
        (at ?o ?to)
  )
 )
)