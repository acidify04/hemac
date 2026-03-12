(define (domain fleet_domain)

(:requirements :strips :typing)

(:types
    quad
    observer
    location
)

(:predicates

    (at_quad ?q - quad ?l - location)
    (at_obs ?o - observer ?l - location)

    (adjacent ?l1 - location ?l2 - location)

    (target ?l - location)
)

(:action move_quad
    :parameters (?q - quad ?from - location ?to - location)

    :precondition (and
        (at_quad ?q ?from)
        (adjacent ?from ?to)
    )

    :effect (and
        (not (at_quad ?q ?from))
        (at_quad ?q ?to)
    )
)

)