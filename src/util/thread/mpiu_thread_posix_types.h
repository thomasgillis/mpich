/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#ifndef MPIU_THREAD_POSIX_TYPES_H_INCLUDED
#define MPIU_THREAD_POSIX_TYPES_H_INCLUDED

#include <errno.h>
#include <pthread.h>
#include "opa_primitives.h"

typedef struct {
    pthread_mutex_t mutex;
    OPA_int_t num_queued_threads;
} MPIU_Thread_mutex_t;
typedef pthread_cond_t MPIU_Thread_cond_t;
typedef pthread_t MPIU_Thread_id_t;
typedef pthread_key_t MPIU_Thread_tls_t;

#define MPIU_THREAD_TLS_T_NULL 0

#endif /* MPIU_THREAD_POSIX_TYPES_H_INCLUDED */
