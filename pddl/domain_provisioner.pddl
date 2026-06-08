(define (domain provisioner-domain)

 (:requirements :typing)

 (:types
    provisioner drone location
 )

 (:predicates
    (at ?p - provisioner ?l - location)
    (at-drone ?d - drone ?l - location)
    (charged ?d - drone)
 )

 (:action move
  :parameters (?p - provisioner ?from - location ?to - location)
  :precondition (and
        (at ?p ?from)
  )
  :effect (and
        (not (at ?p ?from))
        (at ?p ?to)
  )
 )

 (:action recharge-drone
  :parameters (?p - provisioner ?d - drone ?l - location)
  :precondition (and
        (at ?p ?l)
        (at-drone ?d ?l)
  )
  :effect (charged ?d)
 )

)