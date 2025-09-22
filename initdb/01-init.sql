-- Create basic tables structure
CREATE TABLE IF NOT EXISTS "Employee" (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    gender VARCHAR(20),
    "departmentId" INTEGER,
    "roleId" INTEGER,
    "officeId" INTEGER,
    "managerId" INTEGER,
    is_deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS "Department" (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    "organizationId" INTEGER
);

CREATE TABLE IF NOT EXISTS "Role" (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS "Office" (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    "organizationId" INTEGER
);

-- Insert sample data
INSERT INTO "Department" (name, "organizationId") VALUES 
('HR', 1),
('IT', 1),
('Finance', 1);

INSERT INTO "Role" (name) VALUES 
('Manager'),
('Developer'),
('HR Executive');

INSERT INTO "Office" (name, "organizationId") VALUES 
('Dubai Office', 1),
('Abu Dhabi Office', 1);

INSERT INTO "Employee" (name, email, phone, gender, "departmentId", "roleId", "officeId", "managerId") VALUES 
('Faraz', 'faraz@example.com', '+1234567890', 'Male', 1, 1, 1, NULL),
('John Doe', 'john@example.com', '+0987654321', 'Male', 2, 2, 1, 1),
('Jane Smith', 'jane@example.com', '+1122334455', 'Female', 1, 3, 1, 1);
