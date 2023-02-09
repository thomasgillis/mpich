/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef MPIDIG_AM_PART_H_INCLUDED
#define MPIDIG_AM_PART_H_INCLUDED

#include <stdio.h>
#include "ch4_impl.h"
#include "ch4_send.h"
#include "ch4_wait.h"
#include "mpidig_part_utils.h"
#include "mpidpre.h"
#include "mpir_request.h"

int MPIDIG_mpi_psend_init(const void *buf, int partitions, MPI_Aint count,
                          MPI_Datatype datatype, int dest, int tag,
                          MPIR_Comm * comm, MPIR_Info * info, MPIR_Request ** request);
int MPIDIG_mpi_precv_init(void *buf, int partitions, MPI_Aint count,
                          MPI_Datatype datatype, int source, int tag,
                          MPIR_Comm * comm, MPIR_Info * info, MPIR_Request ** request);

MPL_STATIC_INLINE_PREFIX int MPIDIG_part_start(MPIR_Request * request)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    MPID_THREAD_CS_ENTER(VCI, MPIDI_VCI(0).lock);
    MPIR_Assert(!MPIR_Part_request_is_active(request));

    if (request->kind == MPIR_REQUEST_KIND__PART_SEND) {
        /* cc_ptr > 0 indicate data transfer has started and will be completed when cc_ptr = 0
         * the counter depends on the total number of messages, which is know upon reception of the CTS
         * if we have not received the first CTS yet (aka the value of msg_part == -1) we set a temp value of 1*/
        const int msg_part = MPIDIG_PART_REQUEST(request, msg_part);
        MPIR_cc_set(request->cc_ptr, MPL_MAX(1, msg_part));

        /* we have to reset information for the current iteration.
         * The reset is done in the CTS reception callback as well but no msgs has been sent
         * so it's safe to overwrite it*/
        if (MPIDIG_PART_DO_TAG(request)) {
            MPIR_cc_set(&MPIDIG_PART_REQUEST(request, u.send.cc_send), msg_part);
        }
    } else {
        /* cc_ptr > 0 indicate data transfer starts and will be completed when cc_ptr = 0
         * the counter is set to the max of 1 (to avoid early completion) and the number of msg parts
         * that will actually be sent if we have already matched (-1 if not)*/
        MPIR_cc_set(request->cc_ptr, MPL_MAX(1, MPIDIG_PART_REQUEST(request, msg_part)));

        /* if the request has been matched we can use the pointer to check matching status as the
         * pointer is always written inside a lock section */
        const bool is_matched = MPIDIG_Part_rreq_status_has_matched(request);
        if (is_matched) {
            /* we can only allocate now as it's valid to call Precv_init and free immediately (no
             * call to start)*/
            const bool first_cts = MPIDIG_Part_rreq_status_has_first_cts(request);
            if (!first_cts) {
                MPIDIG_Part_rreq_allocate(request);
            }
            MPIR_Assert(MPIDIG_PART_REQUEST(request, msg_part) >= 0);

            const bool do_tag = MPIDIG_PART_DO_TAG(request);
            if (do_tag) {
                /* in tag matching we issue the recv requests we need to remove the lock as the lock
                 * is re-acquired in the recv request creation*/
                MPID_THREAD_CS_EXIT(VCI, MPIDI_VCI(0).lock);
                mpi_errno = MPIDIG_part_issue_recv(request);
                MPID_THREAD_CS_ENTER(VCI, MPIDI_VCI(0).lock);
                MPIR_ERR_CHECK(mpi_errno);

                /* we need to issue the CTS at last to ensure we are fully ready
                 * done only the first time for tag-matching*/
                if (!first_cts) {
                    mpi_errno = MPIDIG_part_issue_cts(request);
                    MPIR_ERR_CHECK(mpi_errno);
                    MPIDIG_Part_rreq_status_first_cts(request);
                }
            } else {
                MPIDIG_part_rreq_reset_cc_part(request);
                mpi_errno = MPIDIG_part_issue_cts(request);
                MPIR_ERR_CHECK(mpi_errno);
                if (!first_cts) {
                    MPIDIG_Part_rreq_status_first_cts(request);
                }
            }
        }
    }

    /* activate must be last to notify that everything has been done */
    MPIR_Part_request_activate(request);

    MPID_THREAD_CS_EXIT(VCI, MPIDI_VCI(0).lock);
  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDIG_mpi_pready_range(int p_low, int p_high,
                                                     MPIR_Request * part_sreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;
    MPIR_Assert(MPIR_Part_request_is_active(part_sreq));
    MPIR_Assert(part_sreq->kind == MPIR_REQUEST_KIND__PART_SEND);

    const int n_part = part_sreq->u.part.partitions;
    MPIR_cc_t *cc_part = MPIDIG_PART_REQUEST(part_sreq, u.send.cc_part);

    /* for each partition mark it as ready and start them if we can */
    for (int i = p_low; i <= p_high; ++i) {
        int incomplete;
        MPIR_cc_decr(&cc_part[i], &incomplete);

        /* send the partition if matched and is complete, try to send the msg */
        if (!incomplete) {
            const int msg_part = MPIDIG_PART_REQUEST(part_sreq, msg_part);
            MPIR_Assert(msg_part >= 0);

            const int msg_lb = MPIDIG_part_idx_lb(i, n_part, msg_part);
#ifndef NDEBUG
            const int msg_ub = MPIDIG_part_idx_ub(i, n_part, msg_part);
            MPIR_Assert(msg_ub - msg_lb == 1);
#endif
            mpi_errno = MPIDIG_part_issue_msg_if_ready(msg_lb, part_sreq, MPIDIG_PART_REGULAR);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            /* if it's not matched or not complete then we miss the CTS for AM or the first CTS for
             * tag send. if we receive the CTS then we will proceed to the send there*/
            MPID_THREAD_CS_ENTER(VCI, MPIDI_VCI(0).lock);
            mpi_errno = MPIDI_progress_test_vci(0);
            MPID_THREAD_CS_EXIT(VCI, MPIDI_VCI(0).lock);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDIG_mpi_pready_list(int length, const int array_of_partitions[],
                                                    MPIR_Request * part_sreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;
    MPIR_Assert(MPIR_Part_request_is_active(part_sreq));
    MPIR_Assert(part_sreq->kind == MPIR_REQUEST_KIND__PART_SEND);

    const int n_part = part_sreq->u.part.partitions;
    MPIR_cc_t *cc_part = MPIDIG_PART_REQUEST(part_sreq, u.send.cc_part);
    for (int ip = 0; ip < length; ip++) {
        const int ipart = array_of_partitions[ip];
        // mark the partition as ready
        int incomplete;
        MPIR_cc_decr(&cc_part[ipart], &incomplete);

        /* send the partition if matched and is complete, try to send the msg */
        if (!incomplete) {
            const int msg_part = MPIDIG_PART_REQUEST(part_sreq, msg_part);
            MPIR_Assert(msg_part >= 0);

            const int msg_lb = MPIDIG_part_idx_lb(ipart, n_part, msg_part);
#ifndef NDEBUG
            const int msg_ub = MPIDIG_part_idx_ub(ipart, n_part, msg_part);
            MPIR_Assert(msg_lb - msg_ub == 1);
#endif
            mpi_errno = MPIDIG_part_issue_msg_if_ready(msg_lb, part_sreq, MPIDIG_PART_REGULAR);
            MPIR_ERR_CHECK(mpi_errno);
        } else {
            /* if it's not matched or not complete then we miss the CTS for AM or the first CTS for
             * tag send. if we receive the CTS then we will proceed to the send there*/
            MPID_THREAD_CS_ENTER(VCI, MPIDI_VCI(0).lock);
            mpi_errno = MPIDI_progress_test_vci(0);
            MPID_THREAD_CS_EXIT(VCI, MPIDI_VCI(0).lock);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDIG_mpi_parrived(MPIR_Request * request, int partition, int *flag)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    /* we must be active to call p_arrived */
    MPIR_Assert(MPIR_Part_request_is_active(request));
    MPIR_Assert(request->kind == MPIR_REQUEST_KIND__PART_RECV);

    int vci = 0;
    /* the request has no cts, check VCI 0 and exit */
    if (unlikely(!MPIDIG_Part_rreq_status_has_first_cts(request))) {
        goto fn_not_arrived;
    }

    /* get the msg to check */
    const int msg_part = MPIDIG_PART_REQUEST(request, msg_part);
    const int n_part = request->u.part.partitions;
    const int msg_id = MPIDIG_part_idx_lb(partition, n_part, msg_part);
#ifndef NDEBUG
    const int end_msg = MPIDIG_part_idx_ub(partition, n_part, msg_part);
    MPIR_Assert(msg_id >= 0);
    MPIR_Assert(end_msg >= 0);
    MPIR_Assert(msg_id <= msg_part);
    MPIR_Assert(end_msg - msg_id == 1);
#endif

    /* it's safe to check do_tag here because we have matched */
    const bool do_tag = MPIDIG_PART_DO_TAG(request);
    if (likely(do_tag)) {
        MPIR_Request *child_req = MPIDIG_PART_RREQUEST(request, tag_req_ptr[msg_id]);
        const bool arrived = MPIR_Request_is_complete(child_req);
        if (arrived) {
            goto fn_arrived;
        } else {
            vci = get_vci_wrapper(child_req);
            goto fn_not_arrived;
        }
    } else {
        MPIR_cc_t *cc_part = MPIDIG_PART_RREQUEST(request, cc_part);
        const bool arrived = (0 == MPIR_cc_get(cc_part[msg_id]));
        if (arrived) {
            goto fn_arrived;
        } else {
            goto fn_not_arrived;
        }
    }
  fn_arrived:
    *flag = TRUE;
    goto fn_exit;
  fn_not_arrived:
    *flag = FALSE;
    /* Trigger progress to process AM packages in case wait with parrived in a loop.
     * also if we don't have the CTS yet we have to receive it -> always on VCI = 0*/
    MPID_THREAD_CS_ENTER(VCI, MPIDI_VCI(vci).lock);
    mpi_errno = MPIDI_progress_test_vci(vci);
    MPID_THREAD_CS_EXIT(VCI, MPIDI_VCI(vci).lock);
    MPIR_ERR_CHECK(mpi_errno);
    goto fn_exit;
  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
#endif /* MPIDIG_AM_PART_H_INCLUDED */
