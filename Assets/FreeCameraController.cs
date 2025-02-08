using UnityEngine;

public class FreeCameraController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float baseMoveSpeed = 10.0f;
    public float fastMoveSpeed = 80.0f;
    public float lookSpeed = 2.0f;
    public float climbSpeed = 10.0f;

    [Header("Rotation Settings")]
    public float yaw = 90.0f;   // You can set an initial yaw here in the Inspector
    public float pitch = 35.0f; // Vertical rotation

    private float moveSpeed;

    void Start()
    {
        moveSpeed = baseMoveSpeed;
        // Optionally, initialize yaw based on the current transform if you want:
        // yaw = transform.eulerAngles.y;
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleCursorLock();
        }

        HandleMovement();
        HandleRotation();
    }

    void HandleMovement()
    {
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            moveSpeed = fastMoveSpeed;
        }
        else
        {
            moveSpeed = baseMoveSpeed;
        }

        float moveForward = Input.GetAxis("Vertical");
        float moveRight = Input.GetAxis("Horizontal");
        float moveUp = 0.0f;

        if (Input.GetKey(KeyCode.Space))
        {
            moveUp += 1.0f;
        }
        if (Input.GetKey(KeyCode.LeftControl))
        {
            moveUp -= 1.0f;
        }

        Vector3 movement = (transform.forward * moveForward + transform.right * moveRight) * moveSpeed * Time.deltaTime;
        movement += transform.up * moveUp * climbSpeed * Time.deltaTime;
        transform.position += movement;
    }

    void HandleRotation()
    {
        // Get mouse input for both axes
        float mouseX = Input.GetAxis("Mouse X") * lookSpeed;
        float mouseY = Input.GetAxis("Mouse Y") * lookSpeed;

        // Update yaw and pitch
        yaw += mouseX;   // This updates the Y axis rotation based on mouse movement.
        pitch -= mouseY;
        pitch = Mathf.Clamp(pitch, -90f, 90f);

        transform.eulerAngles = new Vector3(pitch, yaw, 0.0f);
    }

    void ToggleCursorLock()
    {
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
        else
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }
}
